for FILE in .rfjob/out/*; do cat $FILE; done | awk -F ";" '
    BEGIN{
        PROCINFO["sorted_in"] = "@val_num_desc"
    }
    {
        acc[sprintf("%s;%s", $2, $4)] = $10;
    }
    END{
        count = 1;
        for (i in acc) {
            printf(i";%f\n", acc[i])
            count++;
        }
    }' | head -n 50 #| column --table --table-columns "Dataset_Parameters","RF_Hyperparameters","F1_Score" -s ";"
