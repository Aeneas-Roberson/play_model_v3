# Main training loop
print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)

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

        # Training epoch
        train_losses = []
        num_batches = 10 if TEST_MODE else 100  # Adjust for testing

        for batch_idx in range(num_batches):
            # Create dummy batch (in production, use data iterator)
            batch = dummy_batch

            # Training step
            batch_rng = jax.random.fold_in(rng, state.step)
            state, metrics = train_step(state, batch, batch_rng)
            train_losses.append(metrics['loss'])

            # Log progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {metrics['loss']:.4f}", end='\r')

        # Calculate epoch metrics
        avg_train_loss = np.mean(train_losses)
        training_history['train_loss'].append(avg_train_loss)
        training_history['learning_rate'].append(float(lr_schedule(state.step)))

        # Validation
        val_rmse = None  # Initialize to track RMSE
        if epoch % VALIDATION_FREQUENCY == 0:
            val_losses = []

            for val_batch_idx in range(5 if TEST_MODE else 20):
                val_batch = dummy_batch
                val_rng = jax.random.fold_in(rng, val_batch_idx)
                val_metrics = eval_step(state, val_batch, val_rng)
                val_losses.append(val_metrics['val_loss'])

            avg_val_loss = np.mean(val_losses)
            val_rmse = np.sqrt(avg_val_loss)
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
        if epoch % VALIDATION_FREQUENCY == 0:
            print(f"    Val Loss: {avg_val_loss:.4f}")
            print(f"    Val RMSE: {val_rmse:.4f}")
        print(f"    Time: {epoch_time:.1f}s")
        print(f"    LR: {training_history['learning_rate'][-1]:.6f}")

        # Checkpoint
        if epoch % CHECKPOINT_FREQUENCY == 0:
            checkpoint_metrics = {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss if epoch % VALIDATION_FREQUENCY == 0 else None,
                'val_rmse': val_rmse if val_rmse is not None else None,  # Include RMSE when available
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
print("🎉 TRAINING COMPLETE")
print("="*60)