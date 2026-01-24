  TensorBoard Real-Time Logging Analysis                                                                                                                                               
                                                                                                                                                                                       
  Current State                                                                                                                                                                        
                                                                                                                                                                                       
  src/train.py uses TensorBoard directly via SummaryWriter:                                                                                                                            
  - Lines 675: Creates writer at run start                                                                                                                                             
  - Lines 797-809: Logs epoch-level metrics only (accuracy, loss, coverage, hierarchy)                                                                                                 
  - Line 862: Closes writer at end                                                                                                                                                     
                                                                                                                                                                                       
  src/utils/tensorboard_logger.py has a comprehensive TensorBoardLogger class that is NOT currently used by train.py. It provides:                                                     
  - log_batch() - batch-level loss components (CE, KL)                                                                                                                                 
  - log_hyperbolic_batch() - ranking/radial/centroid losses per batch                                                                                                                  
  - log_hyperbolic_epoch() - full epoch metrics                                                                                                                                        
  - log_epoch() - comprehensive epoch logging                                                                                                                                          
  - log_histograms() - weight/gradient distributions                                                                                                                                   
  - log_manifold_embedding() - 3D latent space visualization                                                                                                                           
                                                                                                                                                                                       
  ---                                                                                                                                                                                  
  Integration Points for Real-Time Logging                                                                                                                                             
                                                                                                                                                                                       
  1. Batch-Level Metrics (inside training loop, ~line 730)                                                                                                                             
                                                                                                                                                                                       
  Location: src/train.py:710-732 (inside for batch_ops, batch_idx loop)                                                                                                                
  Current: Only accumulates loss/acc sums                                                                                                                                              
  Enhancement: Call log_batch() with loss components after each batch                                                                                                                  
  Benefit: Real-time loss curve updates during training                                                                                                                                
                                                                                                                                                                                       
  2. Loss Component Breakdown (~line 721)                                                                                                                                              
                                                                                                                                                                                       
  Location: src/train.py:721-722                                                                                                                                                       
  Current: losses['total'] extracted, components discarded                                                                                                                             
  Enhancement: Log individual loss components (hierarchy, coverage, separation)                                                                                                        
  Benefit: Diagnose which loss is dominating/stuck                                                                                                                                     
                                                                                                                                                                                       
  3. Gradient Statistics (~line 726)                                                                                                                                                   
                                                                                                                                                                                       
  Location: src/train.py:726 (after clip_grad_norm_)                                                                                                                                   
  Current: Gradient norm computed but not logged                                                                                                                                       
  Enhancement: Log gradient norm per batch and per parameter group                                                                                                                     
  Benefit: Detect gradient explosion/vanishing in real-time                                                                                                                            
                                                                                                                                                                                       
  4. Learning Rate Tracking (~line 734)                                                                                                                                                
                                                                                                                                                                                       
  Location: src/train.py:734 (after scheduler.step())                                                                                                                                  
  Current: LR not logged                                                                                                                                                               
  Enhancement: Log current LR from scheduler                                                                                                                                           
  Benefit: Verify warmup/decay behaving correctly                                                                                                                                      
                                                                                                                                                                                       
  5. StateNet Decisions (~line 778-786)                                                                                                                                                
                                                                                                                                                                                       
  Location: src/train.py:778-786                                                                                                                                                       
  Current: StateNet state applied but not logged to TensorBoard                                                                                                                        
  Enhancement: Log freeze/unfreeze events, threshold values, Q_delta                                                                                                                   
  Benefit: Understand when/why components freeze                                                                                                                                       
                                                                                                                                                                                       
  6. Weight Histograms (end of epoch)                                                                                                                                                  
                                                                                                                                                                                       
  Location: After validation (~line 840)                                                                                                                                               
  Current: Not implemented                                                                                                                                                             
  Enhancement: Call log_histograms() every N epochs                                                                                                                                    
  Benefit: Track weight distribution evolution, detect dead neurons                                                                                                                    
                                                                                                                                                                                       
  7. Embedding Visualization (periodic)                                                                                                                                                
                                                                                                                                                                                       
  Location: After validation, every K epochs                                                                                                                                           
  Current: Not implemented                                                                                                                                                             
  Enhancement: Call log_manifold_embedding() periodically                                                                                                                              
  Benefit: Interactive 3D exploration of latent space with 3-adic coloring                                                                                                             
                                                                                                                                                                                       
  ---                                                                                                                                                                                  
  Recommended Changes (Summary)                                                                                                                                                        
  ┌──────────┬─────────┬──────────────────────┬─────────────────┐                                                                                                                      
  │ Location │ Current │     Enhancement      │ Flush Frequency │                                                                                                                      
  ├──────────┼─────────┼──────────────────────┼─────────────────┤                                                                                                                      
  │ Line 730 │ None    │ log_batch()          │ Every batch     │                                                                                                                      
  ├──────────┼─────────┼──────────────────────┼─────────────────┤                                                                                                                      
  │ Line 726 │ None    │ Gradient norm scalar │ Every batch     │                                                                                                                      
  ├──────────┼─────────┼──────────────────────┼─────────────────┤                                                                                                                      
  │ Line 734 │ None    │ LR scalar            │ Every epoch     │                                                                                                                      
  ├──────────┼─────────┼──────────────────────┼─────────────────┤                                                                                                                      
  │ Line 786 │ None    │ StateNet metrics     │ Every eval      │                                                                                                                      
  ├──────────┼─────────┼──────────────────────┼─────────────────┤                                                                                                                      
  │ Line 840 │ None    │ Weight histograms    │ Every 10 epochs │                                                                                                                      
  ├──────────┼─────────┼──────────────────────┼─────────────────┤                                                                                                                      
  │ Line 840 │ None    │ Embeddings           │ Every 50 epochs │                                                                                                                      
  └──────────┴─────────┴──────────────────────┴─────────────────┘                                                                                                                      
  Architecture Recommendation                                                                                                                                                          
                                                                                                                                                                                       
  Replace the direct SummaryWriter usage in train.py with the existing TensorBoardLogger class and verify the SummaryWriter class from train.py and other src components that duplicate logging are properly centralized within TensorBoardLogger:                                                                                        
                                                                                                                                                                                       
  1. Import TensorBoardLogger from src/utils                                                                                                                                           
  2. Initialize at line 675 instead of raw SummaryWriter                                                                                                                               
  3. Call appropriate methods at each integration point                                                                                                                                
  4. Add flush() at end of each epoch for real-time updates                                                                                                                            
                                                                                                                                                                                       
  The TensorBoardLogger class already handles:                                                                                                                                         
  - Null safety (no-op when TensorBoard unavailable)                                                                                                                                   
  - Proper flushing for real-time updates                                                                                                                                              
  - Structured metric naming conventions      
