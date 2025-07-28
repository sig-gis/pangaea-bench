echo Task ${BATCH_TASK_INDEX}

# wandb login 27d91b7cd8af63a77b0f025a0df4e85c3caf7cd5

if [ $BATCH_TASK_INDEX -eq 0 ]; then
    scripts/terramind/limited_label_10pct/terramind_ai4smallfarms_upernet.sh
elif [ $BATCH_TASK_INDEX -eq 1 ]; then
    scripts/terramind/limited_label_10pct/terramind_croptypemapping_upernet.sh
elif [ $BATCH_TASK_INDEX -eq 2 ]; then
    scripts/terramind/limited_label_10pct/terramind_dynamicearthnet_upernet.sh
elif [ $BATCH_TASK_INDEX -eq 3 ]; then
    scripts/terramind/limited_label_10pct/terramind_hlsburnscars_upernet.sh
elif [ $BATCH_TASK_INDEX -eq 4 ]; then
    scripts/terramind/limited_label_10pct/terramind_mados_upernet.sh
elif [ $BATCH_TASK_INDEX -eq 5 ]; then
    scripts/terramind/limited_label_10pct/terramind_mtcropclassification_upernet.sh
elif [ $BATCH_TASK_INDEX -eq 6 ]; then
    scripts/terramind/limited_label_10pct/terramind_pastis_upernet.sh
elif [ $BATCH_TASK_INDEX -eq 7 ]; then
    scripts/terramind/limited_label_10pct/terramind_sen1floods_upernet.sh  
elif [ $BATCH_TASK_INDEX -eq 8 ]; then
    scripts/terramind/limited_label_10pct/terramind_spacenet7_upernet.sh
elif [ $BATCH_TASK_INDEX -eq 9 ]; then
    scripts/terramind/limited_label_10pct/terramind_xview2_upernet.sh
elif [ $BATCH_TASK_INDEX -eq 10 ]; then
    scripts/terramind/limited_label_10pct/terramind_biomasters_upernet.sh
fi