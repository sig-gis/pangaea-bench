echo Task ${BATCH_TASK_INDEX}

# wandb login 27d91b7cd8af63a77b0f025a0df4e85c3caf7cd5

if [ $BATCH_TASK_INDEX -eq 0 ]; then
    scripts/prithvi/test_prithvi_hlsburnscars_linear.sh
elif [ $BATCH_TASK_INDEX -eq 1 ]; then
    scripts/prithvi/test_prithvi_ai4smallfarms_linear.sh
elif [ $BATCH_TASK_INDEX -eq 2 ]; then
    scripts/prithvi/test_prithvi_sen1floods_linear.sh
elif [ $BATCH_TASK_INDEX -eq 3 ]; then
    scripts/scalemae/test_scalemae_hlsburnscars_linear.sh
elif [ $BATCH_TASK_INDEX -eq 4 ]; then
    scripts/scalemae/test_scalemae_ai4smallfarms_linear.sh
elif [ $BATCH_TASK_INDEX -eq 5 ]; then
    scripts/scalemae/test_scalemae_sen1floods_linear.sh
fi

