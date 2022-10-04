for FILE in .rfjob/out/*; do cat $FILE; done | awk -F ":" '
    BEGIN{
        PROCINFO["sorted_in"] = "@val_num_desc"
    }
    {
        acc[sprintf("%s:%s", $2, $4)] = $10;
    }
    END{
        count = 1;
        for (i in acc) {
            printf(i":%f\n", acc[i])
            count++;
        }
    }' | head -n 20 | column --table --table-columns "Dataset Parameters","RF Hyperparameters","F1 Score" -s ":"
