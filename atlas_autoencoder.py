import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Tuple, List, Dict
from itertools import combinations
import seaborn as sns


class BettiEstimator:
    def __init__(self, num_charts: int, tau: float = 0.5):
        self.num_charts = num_charts
        self.tau = tau
        self.overlap_weights = torch.zeros((num_charts, num_charts))
    
    def update_overlap_weights(self, alpha: torch.Tensor):
        """Update overlap weights w_ij = E[min(α_i(x), α_j(x))]"""
        batch_size = alpha.shape[0]
        
        # Compute pairwise minimums for all chart pairs
        for i in range(self.num_charts):
            for j in range(self.num_charts):
                min_alpha = torch.min(alpha[:, i], alpha[:, j])
                self.overlap_weights[i, j] = min_alpha.mean()
    
    def build_nerve_complex(self) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int, int]]]:
        """Build nerve complex N_τ with vertices, edges, and triangles"""
        # Vertices are just chart indices
        vertices = list(range(self.num_charts))
        
        # Edges where w_ij > τ
        edges = []
        for i in range(self.num_charts):
            for j in range(i + 1, self.num_charts):
                if self.overlap_weights[i, j] > self.tau:
                    edges.append((i, j))
        
        # Triangles where all three edges are present
        triangles = []
        edge_set = set(edges)
        for i, j, k in combinations(range(self.num_charts), 3):
            if (i, j) in edge_set and (i, k) in edge_set and (j, k) in edge_set:
                triangles.append((i, j, k))
        
        return vertices, edges, triangles
    
    def build_boundary_operators(self, vertices: List[int], edges: List[Tuple[int, int]], 
                                triangles: List[Tuple[int, int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build boundary operators B_1 and B_2"""
        num_vertices = len(vertices)
        num_edges = len(edges)
        num_triangles = len(triangles)
        
        # B_1: edges -> vertices
        B_1 = torch.zeros((num_edges, num_vertices), dtype=torch.float32)
        for edge_idx, (i, j) in enumerate(edges):
            B_1[edge_idx, i] = -1
            B_1[edge_idx, j] = +1
        
        # B_2: triangles -> edges  
        B_2 = torch.zeros((num_triangles, num_edges), dtype=torch.float32)
        
        # Create edge lookup dict for efficiency
        edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}
        
        for tri_idx, (i, j, k) in enumerate(triangles):
            # Ensure orientation i < j < k
            if not (i < j < k):
                i, j, k = sorted([i, j, k])
            
            # Find edge indices using the lookup dict
            edge_ij = edge_to_idx.get((i, j), -1)
            edge_ik = edge_to_idx.get((i, k), -1)  
            edge_jk = edge_to_idx.get((j, k), -1)
            
            # Set boundary operator entries with correct signs
            if edge_ij >= 0:
                B_2[tri_idx, edge_ij] = +1  # ∂(i,j,k) has +1 coefficient on (i,j)
            if edge_ik >= 0:
                B_2[tri_idx, edge_ik] = -1  # ∂(i,j,k) has -1 coefficient on (i,k) 
            if edge_jk >= 0:
                B_2[tri_idx, edge_jk] = +1  # ∂(i,j,k) has +1 coefficient on (j,k)
        
        return B_1, B_2
    
    def compute_hodge_laplacians(self, B_1: torch.Tensor, B_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Hodge Laplacians Δ_0 and Δ_1"""
        # Δ_0 = B_1^T B_1
        Delta_0 = B_1.T @ B_1
        
        # Δ_1 = B_1 B_1^T + B_2^T B_2
        Delta_1 = B_1 @ B_1.T + B_2.T @ B_2
        
        return Delta_0, Delta_1
    
    def compute_betti_numbers(self, Delta_0: torch.Tensor, Delta_1: torch.Tensor, 
                             eps: float = 1e-6) -> Tuple[int, int]:
        """Compute Betti numbers as multiplicity of zero eigenvalues"""
        # Compute eigenvalues
        eigvals_0 = torch.linalg.eigvals(Delta_0).real
        eigvals_1 = torch.linalg.eigvals(Delta_1).real
        
        # Count eigenvalues close to zero
        b_0 = (eigvals_0.abs() < eps).sum().item()
        b_1 = (eigvals_1.abs() < eps).sum().item()
        
        return b_0, b_1
    
    def estimate_betti_numbers(self, alpha: torch.Tensor, debug: bool = False) -> Tuple[int, int]:
        """Full pipeline: estimate Betti numbers from chart weights"""
        self.update_overlap_weights(alpha)
        vertices, edges, triangles = self.build_nerve_complex()
        
        if debug:
            print(f"Nerve complex: {len(vertices)} vertices, {len(edges)} edges, {len(triangles)} triangles")
            print(f"Overlap weights max: {self.overlap_weights.max():.4f}, mean: {self.overlap_weights.mean():.4f}")
        
        if len(edges) == 0:
            return len(vertices), 0
        
        B_1, B_2 = self.build_boundary_operators(vertices, edges, triangles)
        Delta_0, Delta_1 = self.compute_hodge_laplacians(B_1, B_2)
        
        if debug:
            print(f"Delta_0 shape: {Delta_0.shape}, Delta_1 shape: {Delta_1.shape}")
            eigvals_0 = torch.linalg.eigvals(Delta_0).real
            eigvals_1 = torch.linalg.eigvals(Delta_1).real
            print(f"Delta_0 eigenvalues near zero: {(eigvals_0.abs() < 1e-6).sum()}")
            print(f"Delta_1 eigenvalues near zero: {(eigvals_1.abs() < 1e-6).sum()}")
        
        return self.compute_betti_numbers(Delta_0, Delta_1)
    
    def visualize_nerve_adjacency(self, alpha: torch.Tensor, save_path: str = "nerve_adjacency.png"):
        """Create heatmap visualization of overlap weights and nerve adjacency"""
        self.update_overlap_weights(alpha)
        vertices, edges, triangles = self.build_nerve_complex()
        
        # Create adjacency matrix for nerve complex
        adjacency = torch.zeros((self.num_charts, self.num_charts))
        for i, j in edges:
            adjacency[i, j] = 1
            adjacency[j, i] = 1
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Overlap weights
        sns.heatmap(self.overlap_weights.numpy(), annot=True, fmt='.3f', 
                   cmap='viridis', ax=ax1, cbar_kws={'label': 'Overlap weight'})
        ax1.set_title(f'Chart Overlap Weights\n(τ = {self.tau:.4f})')
        ax1.set_xlabel('Chart j')
        ax1.set_ylabel('Chart i')
        
        # Plot 2: Nerve adjacency
        sns.heatmap(adjacency.numpy(), annot=True, fmt='.0f',
                   cmap='RdBu_r', ax=ax2, cbar_kws={'label': 'Connected'})
        ax2.set_title(f'Nerve Complex Adjacency\n({len(edges)} edges, {len(triangles)} triangles)')
        ax2.set_xlabel('Chart j')
        ax2.set_ylabel('Chart i')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Overlap weights: min={self.overlap_weights.min():.4f}, max={self.overlap_weights.max():.4f}, mean={self.overlap_weights.mean():.4f}")
        print(f"Edges above threshold: {len(edges)}/{self.num_charts*(self.num_charts-1)//2}")
    
    def visualize_nerve_3d(self, model, x: torch.Tensor, alpha: torch.Tensor, save_path: str = "nerve_3d.png"):
        """Create 3D visualization of data colored by charts with nerve complex overlay"""
        self.update_overlap_weights(alpha)
        vertices, edges, triangles = self.build_nerve_complex()
        
        # Get dominant chart assignment for each point
        chart_assignments = torch.argmax(alpha, dim=1)
        
        # Compute chart centers as points where each chart is most active
        chart_centers = torch.zeros((self.num_charts, 3))
        dead_charts = []
        
        for i in range(self.num_charts):
            # Find the point where chart i has maximum activation
            max_activation_idx = torch.argmax(alpha[:, i])
            max_activation = alpha[max_activation_idx, i].item()
            
            if max_activation > 1e-6:  # Chart is active
                # Use the actual data point where this chart is most active
                chart_centers[i] = x[max_activation_idx]
            else:
                dead_charts.append(i)
                chart_centers[i] = torch.tensor([0., 0., 0.])  # Place dead charts at origin
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot data points colored by chart assignment
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_charts))
        for i in range(self.num_charts):
            mask = chart_assignments == i
            if mask.sum() > 0:
                ax.scatter(x[mask, 0], x[mask, 1], x[mask, 2], 
                          c=[colors[i]], alpha=0.6, s=1, label=f'Chart {i}')
        
        # Plot chart centers as larger markers
        for i in range(self.num_charts):
            if i in dead_charts:
                # Plot dead charts as red X at origin
                ax.scatter(chart_centers[i, 0], chart_centers[i, 1], chart_centers[i, 2],
                          c='red', s=200, marker='x', linewidths=3)
                ax.text(chart_centers[i, 0], chart_centers[i, 1], chart_centers[i, 2] + 0.2,
                       f'{i}(DEAD)', fontsize=12, ha='center', weight='bold', color='red')
            else:
                # Plot active charts as black circles
                ax.scatter(chart_centers[i, 0], chart_centers[i, 1], chart_centers[i, 2],
                          c='black', s=100, marker='o', edgecolors='white', linewidths=2)
                ax.text(chart_centers[i, 0], chart_centers[i, 1], chart_centers[i, 2] + 0.2,
                       f'{i}', fontsize=12, ha='center', weight='bold')
        
        # Draw nerve complex edges
        for i, j in edges:
            if (chart_assignments == i).sum() > 0 and (chart_assignments == j).sum() > 0:
                ax.plot([chart_centers[i, 0], chart_centers[j, 0]],
                       [chart_centers[i, 1], chart_centers[j, 1]],
                       [chart_centers[i, 2], chart_centers[j, 2]],
                       'k-', linewidth=2, alpha=0.8)
        
        # Draw nerve complex triangles as filled faces
        if len(triangles) > 0:
            triangle_faces = []
            for i, j, k in triangles:
                # Only draw triangle if all three charts are active
                if ((chart_assignments == i).sum() > 0 and 
                    (chart_assignments == j).sum() > 0 and 
                    (chart_assignments == k).sum() > 0):
                    triangle_face = [
                        [chart_centers[i, 0], chart_centers[i, 1], chart_centers[i, 2]],
                        [chart_centers[j, 0], chart_centers[j, 1], chart_centers[j, 2]],
                        [chart_centers[k, 0], chart_centers[k, 1], chart_centers[k, 2]]
                    ]
                    triangle_faces.append(triangle_face)
            
            if triangle_faces:
                # Add triangular faces with transparency
                face_collection = Poly3DCollection(triangle_faces, alpha=0.3, 
                                                 facecolors='lightblue', edgecolors='blue')
                ax.add_collection3d(face_collection)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title(f'Torus Data with Chart Assignments and Nerve Complex\n'
                    f'{len(edges)} edges, {len(triangles)} triangles, b₀={len(vertices) - len(edges) + len(triangles)}')
        
        # Equal aspect ratio for proper torus shape
        max_range = max(x.max() - x.min() for x in [x[:, 0], x[:, 1], x[:, 2]]) / 2
        mid_x, mid_y, mid_z = x.mean(dim=0)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print chart statistics
        if dead_charts:
            print(f"\nWARNING: Dead charts detected: {dead_charts}")
        
        for i in range(self.num_charts):
            count = (chart_assignments == i).sum().item()
            status = " (DEAD)" if i in dead_charts else ""
            print(f"Chart {i}: {count} points ({100*count/len(x):.1f}%){status}")
        
        return chart_centers, chart_assignments
    
    def visualize_reconstruction_3d(self, x_orig: torch.Tensor, x_recon: torch.Tensor, 
                                   save_path: str = "reconstruction_3d.png"):
        """Create 3D visualization comparing original and reconstructed torus"""
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Reconstruction only
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(x_recon[:, 0], x_recon[:, 1], x_recon[:, 2], 
                   c='darkblue', alpha=0.8, s=2)
        ax1.set_title('Atlas Reconstruction')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Plot 2: Reconstruction error magnitude
        error_mag = torch.norm(x_orig - x_recon, dim=1)
        ax2 = fig.add_subplot(132, projection='3d')
        scatter = ax2.scatter(x_orig[:, 0], x_orig[:, 1], x_orig[:, 2], 
                             c=error_mag.numpy(), cmap='viridis', s=2, alpha=0.8)
        ax2.set_title('Reconstruction Error Magnitude')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        plt.colorbar(scatter, ax=ax2, shrink=0.5, aspect=20)
        
        # Plot 3: Error vectors (subsampled for clarity)
        ax3 = fig.add_subplot(133, projection='3d')
        subsample = slice(0, len(x_orig), max(1, len(x_orig)//500))  # Show ~500 vectors max
        x_sub = x_orig[subsample]
        x_recon_sub = x_recon[subsample]
        
        ax3.scatter(x_sub[:, 0], x_sub[:, 1], x_sub[:, 2], 
                   c='blue', alpha=0.6, s=10, label='Original')
        ax3.scatter(x_recon_sub[:, 0], x_recon_sub[:, 1], x_recon_sub[:, 2], 
                   c='red', alpha=0.6, s=10, label='Reconstruction')
        
        # Draw error vectors
        for i in range(len(x_sub)):
            ax3.plot([x_sub[i, 0], x_recon_sub[i, 0]],
                    [x_sub[i, 1], x_recon_sub[i, 1]], 
                    [x_sub[i, 2], x_recon_sub[i, 2]], 
                    'k-', alpha=0.3, linewidth=0.5)
        
        ax3.set_title('Error Vectors (Subsampled)')
        ax3.legend()
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        # Set equal aspect ratios for all subplots
        for ax in [ax1, ax2, ax3]:
            max_range = max((x_orig.max() - x_orig.min()).item(), 
                           (x_recon.max() - x_recon.min()).item()) / 2
            mid_x = (x_orig[:, 0].mean() + x_recon[:, 0].mean()) / 2
            mid_y = (x_orig[:, 1].mean() + x_recon[:, 1].mean()) / 2
            mid_z = (x_orig[:, 2].mean() + x_recon[:, 2].mean()) / 2
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print reconstruction statistics
        mse = torch.nn.functional.mse_loss(x_recon, x_orig)
        mae = torch.mean(torch.abs(x_recon - x_orig))
        max_error = torch.max(torch.norm(x_orig - x_recon, dim=1))
        mean_error = torch.mean(torch.norm(x_orig - x_recon, dim=1))
        
        print(f"Reconstruction Statistics:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Mean L2 error: {mean_error:.6f}")
        print(f"  Max L2 error: {max_error:.6f}")
        
        return error_mag
    
    def visualize_chart_latents(self, model, x: torch.Tensor, alpha: torch.Tensor, 
                               save_path: str = "chart_latents.png"):
        """Visualize data projected into each chart's local 2D coordinate system"""
        num_charts = model.num_charts
        
        # Calculate grid dimensions for subplots
        cols = min(4, num_charts)
        rows = (num_charts + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        # Ensure axes is always a 2D array for consistent indexing
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Get latent representations for each chart
        with torch.no_grad():
            for i in range(num_charts):
                # Get latent coordinates for this chart
                xi = model.decoders[i](x)  # [N, latent_dim]
                
                # Get chart weights for coloring
                chart_weights = alpha[:, i]
                
                # Calculate subplot position
                row = i // cols
                col = i % cols
                ax = axes[row, col]
                
                # Create scatter plot colored by chart activation
                scatter = ax.scatter(xi[:, 0], xi[:, 1], 
                                   c=chart_weights.numpy(), 
                                   cmap='viridis', s=1, alpha=0.7)
                
                ax.set_title(f'Chart {i} Latent Space\n(colored by activation)')
                ax.set_xlabel('ξ₁')
                ax.set_ylabel('ξ₂')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax, shrink=0.8)
                
                # Print chart statistics
                active_points = (chart_weights > 0.1).sum().item()
                mean_activation = chart_weights.mean().item()
                max_activation = chart_weights.max().item()
                print(f"Chart {i}: {active_points} active points (>0.1), "
                      f"mean α={mean_activation:.3f}, max α={max_activation:.3f}")
        
        # Hide unused subplots
        for i in range(num_charts, rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def visualize_chart_topology(self, model, alpha: torch.Tensor, save_path: str = "chart_topology.png"):
        """Visualize the topographic organization of charts"""
        grid_size = model.chart_grid_size
        positions = model.chart_positions
        
        # Create grid visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Chart usage in grid layout
        chart_usage = alpha.mean(dim=0).numpy()
        usage_grid = np.zeros((grid_size, grid_size))
        
        for i in range(model.num_charts):
            row, col = positions[i].int().tolist()
            if row < grid_size and col < grid_size:
                usage_grid[row, col] = chart_usage[i]
        
        im1 = ax1.imshow(usage_grid, cmap='viridis', interpolation='nearest')
        ax1.set_title('Chart Usage (Topographic Layout)')
        ax1.set_xlabel('Grid Column')
        ax1.set_ylabel('Grid Row')
        
        # Add text annotations
        for i in range(model.num_charts):
            row, col = positions[i].int().tolist()
            if row < grid_size and col < grid_size:
                ax1.text(col, row, f'{i}', ha='center', va='center', 
                        color='white' if chart_usage[i] < chart_usage.max()/2 else 'black',
                        fontsize=8, weight='bold')
        
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Plot 2: Spatial distances between charts
        spatial_distances = torch.cdist(positions, positions).numpy()
        im2 = ax2.imshow(spatial_distances, cmap='plasma', interpolation='nearest')
        ax2.set_title('Spatial Distances Between Charts')
        ax2.set_xlabel('Chart Index')
        ax2.set_ylabel('Chart Index')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Chart grid size: {grid_size}x{grid_size}")
        print(f"Active charts: {(chart_usage > 0.01).sum()}/{model.num_charts}")
        print(f"Chart usage range: [{chart_usage.min():.4f}, {chart_usage.max():.4f}]")


class AtlasAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 3, latent_dim: int = 2, num_charts: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_charts = num_charts
        
        # Create 2D spatial layout for charts (for TopoLoss)
        self.chart_grid_size = int(np.ceil(np.sqrt(num_charts)))
        self.chart_positions = self._create_chart_positions()
        
        # Gating network g: R^D -> R^m
        self.gating = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_charts)
        )
        
        # Chart decoders D_i: M -> R^d (encode to latent)
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, latent_dim)
            ) for _ in range(num_charts)
        ])
        
        # Chart maps C_i: R^d -> M (decode from latent)
        self.charts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim)
            ) for _ in range(num_charts)
        ])
        
        # Betti number estimator
        self.betti_estimator = BettiEstimator(num_charts)
    
    def _create_chart_positions(self):
        """Create 2D grid positions for charts (for TopoLoss)"""
        positions = []
        for i in range(self.num_charts):
            row = i // self.chart_grid_size
            col = i % self.chart_grid_size
            positions.append([row, col])
        return torch.tensor(positions, dtype=torch.float32)
    
    def compute_topo_loss(self, logits: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        Compute TopoLoss on gating network logits to encourage spatial organization.
        
        Args:
            logits: [batch, num_charts] - raw gating network outputs
            sigma: spatial kernel width for distance weighting
        
        Returns:
            TopoLoss value encouraging nearby charts to have similar activations
        """
        batch_size = logits.shape[0]
        
        # Compute pairwise distances between chart positions
        positions = self.chart_positions.to(logits.device)
        pos_diffs = positions.unsqueeze(0) - positions.unsqueeze(1)  # [num_charts, num_charts, 2]
        spatial_distances = torch.norm(pos_diffs, dim=2)  # [num_charts, num_charts]
        
        # Gaussian spatial kernel - closer charts should have more similar activations
        spatial_weights = torch.exp(-spatial_distances**2 / (2 * sigma**2))
        
        # For each sample in batch, compute weighted activation differences
        topo_loss = 0.0
        for b in range(batch_size):
            logits_b = logits[b]  # [num_charts]
            
            # Compute pairwise differences in activations
            activation_diffs = (logits_b.unsqueeze(0) - logits_b.unsqueeze(1))**2  # [num_charts, num_charts]
            
            # Weight by spatial proximity - penalize differences between nearby charts
            weighted_diffs = spatial_weights * activation_diffs
            
            # Sum over all chart pairs (exclude diagonal)
            mask = torch.eye(self.num_charts, device=logits.device) == 0
            topo_loss += weighted_diffs[mask].sum()
        
        return topo_loss / batch_size
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        
        # Compute soft chart weights α_i = softmax(g(x))
        logits = self.gating(x)  # [batch, num_charts]
        alpha = F.softmax(logits, dim=1)  # [batch, num_charts]
        
        # For each chart: ξ_i = D_i(x), x̂_i = C_i(ξ_i)
        reconstructions = []
        for i in range(self.num_charts):
            xi = self.decoders[i](x)  # [batch, latent_dim]
            x_hat_i = self.charts[i](xi)  # [batch, input_dim]
            reconstructions.append(x_hat_i)
        
        # Stack and weight: x̂ = Σ α_i x̂_i
        x_hat_stack = torch.stack(reconstructions, dim=2)  # [batch, input_dim, num_charts]
        alpha_expanded = alpha.unsqueeze(1)  # [batch, 1, num_charts]
        x_hat = torch.sum(x_hat_stack * alpha_expanded, dim=2)  # [batch, input_dim]
        
        return x_hat, alpha, logits
    
    def get_betti_numbers(self, alpha: torch.Tensor) -> Tuple[int, int]:
        """Estimate Betti numbers from current chart activations"""
        return self.betti_estimator.estimate_betti_numbers(alpha)


def generate_torus_data(n_samples: int = 10000, R: float = 2.0, r: float = 1.0) -> torch.Tensor:
    """Generate points uniformly on torus surface with major radius R, minor radius r."""
    # Sample angles uniformly
    u = torch.rand(n_samples) * 2 * np.pi  # [0, 2π]
    v = torch.rand(n_samples) * 2 * np.pi  # [0, 2π]
    
    # Torus parametrization
    x = (R + r * torch.cos(v)) * torch.cos(u)
    y = (R + r * torch.cos(v)) * torch.sin(u)
    z = r * torch.sin(v)
    
    return torch.stack([x, y, z], dim=1)


def compute_loss(model, x_hat: torch.Tensor, x: torch.Tensor, alpha: torch.Tensor, logits: torch.Tensor,
                 lambda_ent: float = -0.5, lambda_topo: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reconstruction, entropy, and topographic losses."""
    # Reconstruction loss
    loss_rec = F.mse_loss(x_hat, x)
    
    # Entropy regularization: -Σ α_i log α_i
    loss_ent = -torch.sum(alpha * torch.log(alpha + 1e-8), dim=1).mean()
    
    # TopoLoss on gating network output
    loss_topo = model.compute_topo_loss(logits)
    
    # Total loss
    loss_total = loss_rec + lambda_ent * loss_ent + lambda_topo * loss_topo
    
    return loss_total, loss_rec, loss_ent, loss_topo


def train_atlas_autoencoder():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AtlasAutoencoder(input_dim=3, latent_dim=2, num_charts=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate torus data
    train_data = generate_torus_data(n_samples=10000).to(device)
    
    # Training loop
    model.train()
    for epoch in range(1001):
        optimizer.zero_grad()
        
        x_hat, alpha, logits = model(train_data)
        loss_total, loss_rec, loss_ent, loss_topo = compute_loss(model, x_hat, train_data, alpha, logits)
        
        loss_total.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            # Compute Betti numbers
            with torch.no_grad():
                debug_flag = epoch == 11000  # Debug on final epoch
                b_0, b_1 = model.betti_estimator.estimate_betti_numbers(alpha.cpu(), debug=debug_flag)
            print(f"Epoch {epoch}: Total={loss_total:.4f}, Rec={loss_rec:.4f}, Ent={loss_ent:.4f}, Topo={loss_topo:.4f}, b₀={b_0}, b₁={b_1}")
    
    return model, train_data


if __name__ == "__main__":
    model, data = train_atlas_autoencoder()
    
    # Test reconstruction
    model.eval()
    with torch.no_grad():
        # Use a subset for reconstruction test
        test_data = data[:1000]
        x_hat, alpha, _ = model(test_data)
        rec_error = F.mse_loss(x_hat, test_data)
        print(f"Final reconstruction error: {rec_error:.6f}")
        
        # Get alpha for full dataset for visualization
        _, alpha_full, _ = model(data)
        
        # Show chart usage
        chart_usage = alpha_full.mean(dim=0)
        print(f"Chart usage: {chart_usage.cpu().numpy()}")
        
        # Visualize nerve adjacency
        print("\nGenerating nerve adjacency heatmap...")
        model.betti_estimator.visualize_nerve_adjacency(alpha_full.cpu())
        
        # Visualize 3D nerve complex on torus
        print("\nGenerating 3D nerve visualization...")
        chart_centers, assignments = model.betti_estimator.visualize_nerve_3d(model, data.cpu(), alpha_full.cpu())
        
        # Visualize reconstruction quality
        print("\nGenerating reconstruction comparison...")
        x_recon_full, _, _ = model(data)
        error_mag = model.betti_estimator.visualize_reconstruction_3d(data.cpu(), x_recon_full.cpu())
        
        # Visualize chart latent spaces
        print("\nGenerating chart latent space visualizations...")
        model.betti_estimator.visualize_chart_latents(model, data.cpu(), alpha_full.cpu())
        
        # Visualize topographic organization
        print("\nGenerating chart topology visualization...")
        model.betti_estimator.visualize_chart_topology(model, alpha_full.cpu()) 