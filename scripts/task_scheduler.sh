echo Task ${BATCH_TASK_INDEX}

wandb login 27d91b7cd8af63a77b0f025a0df4e85c3caf7cd5

if [ $BATCH_TASK_INDEX -eq 0 ]; then
    scripts/dofa/test_dofa_hlsburnscars_muster.sh
elif [ $BATCH_TASK_INDEX -eq 1 ]; then
    scripts/dofa/test_dofa_ai4smallfarms_muster.sh
elif [ $BATCH_TASK_INDEX -eq 2 ]; then
    scripts/dofa/test_dofa_mados_muster.sh
elif [ $BATCH_TASK_INDEX -eq 3 ]; then
    scripts/dofa/test_dofa_sen1floods_muster.sh
elif [ $BATCH_TASK_INDEX -eq 4 ]; then
    scripts/dofa/test_dofa_spacenet7_muster.sh
elif [ $BATCH_TASK_INDEX -eq 5 ]; then
    scripts/prithvi/test_prithvi_hlsburnscars_muster.sh
elif [ $BATCH_TASK_INDEX -eq 6 ]; then
    scripts/prithvi/test_prithvi_mados_muster.sh
elif [ $BATCH_TASK_INDEX -eq 7 ]; then
    scripts/prithvi/test_prithvi_sen1floods_muster.sh
elif [ $BATCH_TASK_INDEX -eq 8 ]; then
    scripts/prithvi/test_prithvi_ai4smallfarms_muster.sh
fi