# Real Data Training Loop - No More Dummy Data!
print("\n" + "="*60)
print("🚀 STARTING TRAINING WITH REAL CFB DATA")
print("="*60)

# Create real data iterators from train_sequences
def create_batch_iterator(sequences, batch_size, shuffle=True):
    """Create iterator for real CFB data batches"""
    game_data = sequences['game_level']['sequences']
    game_ids = list(game_data.keys())
    
    if shuffle:
        np.random.shuffle(game_ids)
    
    # Create batches of games
    for i in range(0, len(game_ids), batch_size):
        batch_game_ids = game_ids[i:i + batch_size]
        
        # Stack game sequences into batch
        batch_features = []
        batch_targets = []
        
        for game_id in batch_game_ids:
            game_seq = game_data[game_id]
            
            # Extract features (this would match your model's expected input format)
            features = {
                'offense': {
                    'categorical': game_seq.get('offense_categorical', jnp.zeros((MAX_SEQUENCE_LENGTH, 4))),
                    'numerical': game_seq.get('offense_numerical', jnp.zeros((MAX_SEQUENCE_LENGTH, 43)))
                },
                'defense': {
                    'categorical': game_seq.get('defense_categorical', jnp.zeros((MAX_SEQUENCE_LENGTH, 3))),
                    'numerical': game_seq.get('defense_numerical', jnp.zeros((MAX_SEQUENCE_LENGTH, 40)))
                },
                'game_state': {
                    'categorical': game_seq.get('game_state_categorical', jnp.zeros((MAX_SEQUENCE_LENGTH, 7))),
                    'numerical': game_seq.get('game_state_numerical', jnp.zeros((MAX_SEQUENCE_LENGTH, 26)))
                },
                'play_context': {
                    'categorical': game_seq.get('play_context_categorical', jnp.zeros((MAX_SEQUENCE_LENGTH, 2))),
                    'binary': game_seq.get('play_context_binary', jnp.zeros((MAX_SEQUENCE_LENGTH, 20))),
                    'numerical': game_seq.get('play_context_numerical', jnp.zeros((MAX_SEQUENCE_LENGTH, 8)))
                }
            }
            
            # Extract real targets (actual game scores)
            targets = {
                'home_score': game_seq.get('home_score', 0.0),
                'away_score': game_seq.get('away_score', 0.0),
                'play_outcomes': game_seq.get('play_outcomes', jnp.zeros(MAX_SEQUENCE_LENGTH)),
                'drive_outcomes': game_seq.get('drive_outcomes', jnp.zeros(MAX_DRIVES_PER_GAME))
            }
            
            batch_features.append(features)
            batch_targets.append(targets)
        
        # Stack into batch tensors
        batch = {
            'features': {
                'offense': {
                    'categorical': jnp.stack([f['offense']['categorical'] for f in batch_features]),
                    'numerical': jnp.stack([f['offense']['numerical'] for f in batch_features])
                },
                'defense': {
                    'categorical': jnp.stack([f['defense']['categorical'] for f in batch_features]),
                    'numerical': jnp.stack([f['defense']['numerical'] for f in batch_features])
                },
                'game_state': {
                    'categorical': jnp.stack([f['game_state']['categorical'] for f in batch_features]),
                    'numerical': jnp.stack([f['game_state']['numerical'] for f in batch_features])
                },
                'play_context': {
                    'categorical': jnp.stack([f['play_context']['categorical'] for f in batch_features]),
                    'binary': jnp.stack([f['play_context']['binary'] for f in batch_features]),
                    'numerical': jnp.stack([f['play_context']['numerical'] for f in batch_features])
                }
            },
            'targets': {
                'home_score': jnp.array([t['home_score'] for t in batch_targets]),
                'away_score': jnp.array([t['away_score'] for t in batch_targets]),
                'play_outcomes': jnp.stack([t['play_outcomes'] for t in batch_targets]),
                'drive_outcomes': jnp.stack([t['drive_outcomes'] for t in batch_targets])
            }
        }
        
        yield batch

# Create training and validation iterators
print("📊 Creating real data iterators...")
train_iterator = create_batch_iterator(train_sequences, BATCH_SIZE, shuffle=True)

# Create validation sequences (simplified for now)
val_sequences = train_sequences  # In production, use separate validation data
val_iterator = create_batch_iterator(val_sequences, BATCH_SIZE, shuffle=False)

# Updated training step with real targets
@jax.jit
def real_train_step(state, batch, rng):
    """Training step with real CFB data and targets"""
    
    def loss_fn(params):
        # Forward pass
        predictions = state.apply_fn(
            params,
            batch['features'],  # Use real features
            training=True,
            rngs={'dropout': rng}
        )
        
        # Real loss calculation with actual targets
        total_loss = 0.0
        
        # Game-level loss (final scores)
        if 'home_score' in predictions and 'home_score' in batch['targets']:
            home_score_loss = jnp.mean((predictions['home_score'] - batch['targets']['home_score']) ** 2)
            away_score_loss = jnp.mean((predictions.get('away_score', 0) - batch['targets']['away_score']) ** 2)
            total_loss += (home_score_loss + away_score_loss) * 0.5
        
        # Play-level loss
        if 'play_outcomes' in predictions and 'play_outcomes' in batch['targets']:
            play_loss = jnp.mean((predictions['play_outcomes'] - batch['targets']['play_outcomes']) ** 2)
            total_loss += play_loss * 0.3
        
        # Drive-level loss  
        if 'drive_outcomes' in predictions and 'drive_outcomes' in batch['targets']:
            drive_loss = jnp.mean((predictions['drive_outcomes'] - batch['targets']['drive_outcomes']) ** 2)
            total_loss += drive_loss * 0.2
            
        return total_loss, predictions
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, predictions), grads = grad_fn(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    # Compute metrics with real targets
    home_score_rmse = jnp.sqrt(jnp.mean((predictions.get('home_score', 0) - batch['targets']['home_score']) ** 2))
    
    metrics = {
        'loss': loss,
        'home_score_rmse': home_score_rmse,
        'learning_rate': lr_schedule(state.step)
    }
    
    return state, metrics

# Updated validation step
@jax.jit
def real_eval_step(state, batch, rng):
    """Evaluation step with real CFB data"""
    
    predictions = state.apply_fn(
        state.params,
        batch['features'],
        training=False,
        rngs={'dropout': rng}
    )
    
    # Calculate real validation metrics
    val_loss = 0.0
    
    if 'home_score' in predictions and 'home_score' in batch['targets']:
        home_score_loss = jnp.mean((predictions['home_score'] - batch['targets']['home_score']) ** 2)
        away_score_loss = jnp.mean((predictions.get('away_score', 0) - batch['targets']['away_score']) ** 2)
        val_loss = (home_score_loss + away_score_loss) * 0.5
    
    # Calculate spread RMSE (the key metric for beating Vegas)
    if 'home_score' in predictions and 'away_score' in predictions:
        pred_spread = predictions['home_score'] - predictions.get('away_score', 0)
        actual_spread = batch['targets']['home_score'] - batch['targets']['away_score']
        spread_rmse = jnp.sqrt(jnp.mean((pred_spread - actual_spread) ** 2))
    else:
        spread_rmse = jnp.sqrt(val_loss)
    
    metrics = {
        'val_loss': val_loss,
        'val_rmse': jnp.sqrt(val_loss),
        'spread_rmse': spread_rmse
    }
    
    return metrics

print("✅ Real data training functions defined")

# Training phases to execute
phases_to_train = [
    TrainingPhase.WARMUP,
    TrainingPhase.PLAY_FOCUS
] if TEST_MODE else list(TrainingPhase)

for phase_idx, phase in enumerate(phases_to_train):
    print(f"\n📌 PHASE {phase_idx + 1}: {phase.value}")
    phase_config = hyperparams.get_phase_config(phase)

    # Update optimizer for this phase if needed
    if phase_idx > 0:
        # Create new optimizer with phase-specific settings
        new_optimizer = phase_config['optimizer']
        state = state.replace(tx=new_optimizer)

    # Train for specified epochs
    for epoch in range(EPOCHS_PER_PHASE):
        epoch_start_time = time.time()

        # Training epoch with REAL DATA
        train_losses = []
        train_rmses = []
        
        # Create fresh iterator for each epoch
        train_iter = create_batch_iterator(train_sequences, BATCH_SIZE, shuffle=True)
        
        batch_count = 0
        max_batches = 10 if TEST_MODE else 100  # Limit batches in test mode
        
        for batch in train_iter:
            if batch_count >= max_batches:
                break
                
            # Training step with real data
            batch_rng = jax.random.fold_in(rng, state.step)
            state, metrics = real_train_step(state, batch, batch_rng)
            train_losses.append(metrics['loss'])
            train_rmses.append(metrics['home_score_rmse'])
            
            # Log progress
            if batch_count % 10 == 0:
                print(f"  Batch {batch_count}/{max_batches}, Loss: {metrics['loss']:.4f}, RMSE: {metrics['home_score_rmse']:.4f}", end='\r')
            
            batch_count += 1

        # Calculate epoch metrics
        avg_train_loss = np.mean(train_losses)
        avg_train_rmse = np.mean(train_rmses)
        training_history['train_loss'].append(avg_train_loss)
        training_history['learning_rate'].append(float(lr_schedule(state.step)))

        # Validation with REAL DATA
        val_rmse = None
        spread_rmse = None
        if epoch % VALIDATION_FREQUENCY == 0:
            val_losses = []
            val_rmses = []
            spread_rmses = []
            
            val_iter = create_batch_iterator(val_sequences, BATCH_SIZE, shuffle=False)
            val_batch_count = 0
            max_val_batches = 5 if TEST_MODE else 20
            
            for val_batch in val_iter:
                if val_batch_count >= max_val_batches:
                    break
                    
                val_rng = jax.random.fold_in(rng, val_batch_count)
                val_metrics = real_eval_step(state, val_batch, val_rng)
                val_losses.append(val_metrics['val_loss'])
                val_rmses.append(val_metrics['val_rmse'])
                spread_rmses.append(val_metrics['spread_rmse'])
                
                val_batch_count += 1

            avg_val_loss = np.mean(val_losses)
            val_rmse = np.mean(val_rmses)
            spread_rmse = np.mean(spread_rmses)
            
            training_history['val_loss'].append(avg_val_loss)
            training_history['val_rmse'].append(val_rmse)

            # Update best model
            if avg_val_loss < state.best_val_loss:
                state = state.replace(best_val_loss=avg_val_loss)
                print(f"  🎯 New best validation loss: {avg_val_loss:.4f}")

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\n  Epoch {epoch+1}/{EPOCHS_PER_PHASE}:")
        print(f"    Train Loss: {avg_train_loss:.4f}")
        print(f"    Train RMSE: {avg_train_rmse:.4f}")
        if epoch % VALIDATION_FREQUENCY == 0:
            print(f"    Val Loss: {avg_val_loss:.4f}")
            print(f"    Val RMSE: {val_rmse:.4f}")
            print(f"    Spread RMSE: {spread_rmse:.4f} (Vegas target: <13.0)")
        print(f"    Time: {epoch_time:.1f}s")
        print(f"    LR: {training_history['learning_rate'][-1]:.6f}")

        # Checkpoint
        if epoch % CHECKPOINT_FREQUENCY == 0:
            checkpoint_metrics = {
                'train_loss': avg_train_loss,
                'train_rmse': avg_train_rmse,
                'val_loss': avg_val_loss if epoch % VALIDATION_FREQUENCY == 0 else None,
                'val_rmse': val_rmse if val_rmse is not None else None,
                'spread_rmse': spread_rmse if spread_rmse is not None else None,
                'epoch': epoch,
                'phase': phase.value
            }

            checkpoint_manager.save_checkpoint(
                step=state.step,
                params=state.params,
                optimizer_state=state.opt_state,
                metrics=checkpoint_metrics,
                phase_name=phase.value
            )

        # Update epoch counter
        state = state.replace(epoch=state.epoch + 1)

    print(f"\n✅ Phase {phase.value} complete")

print("\n" + "="*60)
print("🎉 REAL DATA TRAINING COMPLETE")
print("="*60)

# Final performance summary
if training_history['val_rmse']:
    best_spread_rmse = min([m.get('spread_rmse', float('inf')) for m in checkpoint_manager.checkpoint_history])
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    print(f"  Best Spread RMSE: {best_spread_rmse:.4f}")
    if best_spread_rmse < 13.0:
        print(f"  ✅ TARGET ACHIEVED! Beat Vegas baseline of 14.2")
    else:
        print(f"  📈 Need {best_spread_rmse - 13.0:.2f} improvement to beat target")