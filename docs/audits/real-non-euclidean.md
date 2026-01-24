● Here's the complete list of deep architectural changes needed:                                                                                                                       
                                                                                                                                                                                       
  1. VAE Sampling (vae.py:220-224)                                                                                                                                                     
                                                                                                                                                                                       
  Current: Euclidean reparameterization                                                                                                                                                
  def reparameterize(self, mu, logvar):                                                                                                                                                
      std = torch.exp(0.5 * logvar)                                                                                                                                                    
      eps = torch.randn_like(std)                                                                                                                                                      
      return mu + eps * std  # Euclidean                                                                                                                                               
                                                                                                                                                                                       
  Required for fully-hyperbolic:                                                                                                                                                       
  def reparameterize(self, mu, logvar, manifold=None):                                                                                                                                 
      std = torch.exp(0.5 * logvar)                                                                                                                                                    
      eps = torch.randn_like(std)                                                                                                                                                      
      z_tangent = mu + eps * std  # Sample in tangent space T₀M                                                                                                                        
                                                                                                                                                                                       
      if manifold is not None:  # Fully hyperbolic mode                                                                                                                                
          return manifold.expmap0(z_tangent)  # Wrapped normal → manifold                                                                                                              
      return z_tangent  # Euclidean mode                                                                                                                                               
                                                                                                                                                                                       
  ---                                                                                                                                                                                  
  2. Decoder Input (vae.py:245-247)                                                                                                                                                    
                                                                                                                                                                                       
  Current: Decodes from Euclidean z (WRONG for hyperbolic)                                                                                                                             
  logits_A = self.decoder_A(z_A_euc)  # Ignores z_hyp entirely!                                                                                                                        
                                                                                                                                                                                       
  Required:                                                                                                                                                                            
  if self.geometry_mode == FULLY_HYPERBOLIC:                                                                                                                                           
      z_A_tangent = self.manifold.logmap0(z_A_hyp)  # Back to tangent space                                                                                                            
      logits_A = self.decoder_A(z_A_tangent)                                                                                                                                           
  else:                                                                                                                                                                                
      logits_A = self.decoder_A(z_A_euc)                                                                                                                                               
                                                                                                                                                                                       
  ---                                                                                                                                                                                  
  3. Projection Layer (hyperbolic_projection.py:182-188)                                                                                                                               
                                                                                                                                                                                       
  Current: Euclidean direction × radius                                                                                                                                                
  direction = F.normalize(z_euclidean + direction_residual, dim=-1)                                                                                                                    
  radius = self.radius_net(z_euclidean) * self.max_radius                                                                                                                              
  z_hyp = direction * radius  # NOT using expmap!                                                                                                                                      
                                                                                                                                                                                       
  Required for fully-hyperbolic:                                                                                                                                                       
  if self.geometry_mode == FULLY_HYPERBOLIC:                                                                                                                                           
      # z_euclidean IS tangent vector, use expmap directly                                                                                                                             
      z_hyp = self.manifold.expmap0(z_euclidean)                                                                                                                                       
  else:                                                                                                                                                                                
      # Current projection approach                                                                                                                                                    
      z_hyp = direction * radius                                                                                                                                                       
                                                                                                                                                                                       
  ---                                                                                                                                                                                  
  4. ManifoldParameter for Riemannian Gradients                                                                                                                                        
                                                                                                                                                                                       
  Current: z_hyp is a regular Tensor                                                                                                                                                   
  Required: Wrap as ManifoldParameter for RiemannianAdam to work:                                                                                                                      
                                                                                                                                                                                       
  from geoopt import ManifoldParameter                                                                                                                                                 
                                                                                                                                                                                       
  # In forward pass, when returning z_hyp:                                                                                                                                             
  z_hyp = ManifoldParameter(z_hyp, manifold=self.manifold)                                                                                                                             
                                                                                                                                                                                       
  Without this, RiemannianAdam sees regular tensors and falls back to Euclidean updates.                                                                                               
                                                                                                                                                                                       
  ---                                                                                                                                                                                  
  5. KL Divergence (NEW - doesn't exist yet)                                                                                                                                           
                                                                                                                                                                                       
  Current: No KL term (or implicit Euclidean KL)                                                                                                                                       
  Required for hyperbolic: Hyperbolic KL divergence                                                                                                                                    
                                                                                                                                                                                       
  def hyperbolic_kl_divergence(mu, logvar, manifold):                                                                                                                                  
      """KL divergence for wrapped normal on Poincaré ball."""                                                                                                                         
      # See Mathieu et al. 2019 "Continuous Hierarchical Representations"                                                                                                              
      # KL(q(z|x) || p(z)) where p(z) is wrapped normal at origin                                                                                                                      
                                                                                                                                                                                       
      var = torch.exp(logvar)                                                                                                                                                          
      # Closed form for wrapped normal KL:                                                                                                                                             
      lambda_mu = manifold.lambda_x(mu)  # Conformal factor                                                                                                                            
      kl = 0.5 * (var * lambda_mu.pow(2) + mu.pow(2).sum(-1) - logvar.sum(-1) - mu.size(-1))                                                                                           
      return kl.mean()                                                                                                                                                                 
                                                                                                                                                                                       
  ---                                                                                                                                                                                  
  6. Geodesic Interpolation (for generation/visualization)                                                                                                                             
                                                                                                                                                                                       
  Current: Linear interpolation (WRONG)                                                                                                                                                
  z_interp = z1 + t * (z2 - z1)  # Euclidean                                                                                                                                           
                                                                                                                                                                                       
  Required:                                                                                                                                                                            
  z_interp = manifold.geodesic(t, z1, z2)  # True geodesic path                                                                                                                        
                                                                                                                                                                                       
  ---                                                                                                                                                                                  
  7. Parallel Transport (for sequential operations)                                                                                                                                    
                                                                                                                                                                                       
  If moving vectors between tangent spaces:                                                                                                                                            
  # Transport vector v from T_x M to T_y M                                                                                                                                             
  v_transported = manifold.transp(x, y, v)                                                                                                                                             
                                                                                                                                                                                       
  ---                                                                                                                                                                                  
  8. Config Schema Update                                                                                                                                                              
                                                                                                                                                                                       
  geometry:                                                                                                                                                                            
    mode: "fully_hyperbolic"  # or "euclidean_projected"                                                                                                                               
    curvature: 1.0                                                                                                                                                                     
    learnable_curvature: false                                                                                                                                                         
                                                                                                                                                                                       
  riemannian:                                                                                                                                                                          
    enabled: true  # MUST be true for fully_hyperbolic                                                                                                                                 
                                                                                                                                                                                       
  ---                                                                                                                                                                                  
  Summary: Files to Modify                                                                                                                                                             
  ┌─────────────────────────────────────┬──────────────────────────────────────────────────────────────────┐                                                                           
  │                File                 │                             Changes                              │                                                                           
  ├─────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                                           
  │ src/models/vae.py                   │ Add geometry_mode, wrapped normal sampling, decoder uses logmap0 │                                                                           
  ├─────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                                           
  │ src/models/hyperbolic_projection.py │ Use expmap0 in fully_hyperbolic mode                             │                                                                           
  ├─────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                                           
  │ src/losses/combined.py              │ Add hyperbolic KL divergence term                                │                                                                           
  ├─────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                                           
  │ src/geometry/poincare.py            │ Export expmap0, logmap0, geodesic, transp wrappers               │                                                                           
  ├─────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                                           
  │ src/train.py                        │ Enforce RiemannianAdam when fully_hyperbolic                     │                                                                           
  ├─────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤                                                                           
  │ src/presets/*.yaml                  │ Add geometry.mode field                                          │                                                                           
  └─────────────────────────────────────┴──────────────────────────────────────────────────────────────────┘                                                                           
  The key insight: expmap0/logmap0 are the bridge between your Euclidean MLPs and the hyperbolic manifold. Without them, you're just using hyperbolic distance as a loss signal on     
  Euclidean embeddings. 