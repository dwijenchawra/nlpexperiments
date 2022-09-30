for FILE in .job/out/*; do cat $FILE; done | awk -F ":" '
    BEGIN{
        PROCINFO["sorted_in"] = "@val_num_desc"
    }
    {
        acc[$2] = $4;  
    }
    END{
        count = 1;
        for (i in acc) {
            printf(i":%f\n", acc[i])
            count++;
        }
    }' | head -n 20 | column --table --table-columns "Hyperparameters","Test Accuracy" -s ":"