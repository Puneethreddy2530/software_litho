# Phase 3: Class & Dataset Distribution Audit\n\n## Methodology\nCalculated exact image dimensions, file size in KB, and visual statistics (Mean RGB, Std RGB, Brightness) across all files in both DCID variants.\n\n## General Verified Facts\n- **Dimensions**: OBSERVED that 100% of images are 512x512.\n\n## Class Level Aggregates (DCID-512-7)\n\n**OBSERVED distributional differences:**\n- Inter-class Brightness Std: 25.25\n- Inter-class File Size Std: 6.83 KB\n\n*(Note: These statistics represent observed differences and do not prove shortcut learning.)*\n\n| Class             |   Size KB (Mean) |   Size KB (Std) |   Brightness (Mean) |   Brightness (Std) |   Count |
|:------------------|-----------------:|----------------:|--------------------:|-------------------:|--------:|
| 1.Red sandstone   |          48.6428 |         4.62021 |              90.043 |            17.2193 |    5000 |
| 2.Light sandstone |          37.5826 |         5.91277 |             139.7   |            16.6499 |    5000 |
| 3.Gray siltstone  |          38.6499 |         3.73976 |             125.031 |            16.9935 |    5000 |
| 4.Mudstone        |          27.7085 |         6.27634 |             100.106 |            10.2455 |    5000 |
| 5.Granite         |          44.1882 |         2.42723 |             159.649 |            16.617  |    5000 |
| 6.Basalt          |          35.0268 |         2.59222 |             103.186 |            10.373  |    5000 |
| 7.Marble          |          34.4033 |         6.38555 |             137.213 |            29.1724 |    5000 |\n\n## Class Level Aggregates (DCID-512-35)\n\n**OBSERVED distributional differences:**\n- Inter-class Brightness Std: 36.17\n- Inter-class File Size Std: 9.10 KB\n\n*(Note: These statistics represent observed differences and do not prove shortcut learning.)*\n\n| Class                       |   Size KB (Mean) |   Size KB (Std) |   Brightness (Mean) |   Brightness (Std) |   Count |
|:----------------------------|-----------------:|----------------:|--------------------:|-------------------:|--------:|
| 1.Red sandstone             |          50.0323 |        3.56459  |             81.9765 |           12.092   |    1000 |
| 10.Monzogranite             |          60.0616 |        3.04532  |            164.378  |           20.4515  |    1000 |
| 11.Fine-grained granite     |          44.4813 |        2.67054  |            179.087  |           10.0986  |    1000 |
| 12.Pegmatite                |          46.4909 |        2.95585  |            129.884  |           17.8654  |    1000 |
| 13.Basalt                   |          34.8776 |        2.39756  |            107.043  |            7.7862  |    1000 |
| 14.Fine-grained basalt      |          48.1788 |        1.78945  |             79.7988 |           11.0384  |    1000 |
| 15.Gabbro                   |          41.9843 |        2.84053  |             96.4162 |            9.25492 |    1000 |
| 16.Dolerite                 |          51.7879 |        2.32337  |            124.836  |           13.6177  |    1000 |
| 17.White marble             |          31.4788 |        3.14452  |            170.232  |           18.95    |    1000 |
| 18.Blue marble              |          27.666  |        2.92811  |            104.316  |           14.9246  |    1000 |
| 19.Dolomite                 |          37.4001 |        1.65338  |            183.485  |           15.1662  |    1000 |
| 2.Light sandstone           |          44.8899 |        2.4077   |            159.149  |           11.6145  |    1000 |
| 20.Black Shale              |          32.7379 |        2.03606  |             94.9623 |           19.1814  |    1000 |
| 21.Red shale                |          30.2193 |        2.16219  |             34.5627 |            8.36086 |    1000 |
| 22.Dark green calc-silicate |          31.1238 |        2.56663  |            114.269  |           12.3362  |    1000 |
| 23.Brown calc-silicate      |          56.2939 |        2.22334  |            137.348  |           19.1472  |    1000 |
| 24.Skarn                    |          35.3691 |        2.04483  |            103.445  |            8.65988 |    1000 |
| 25.Red-brown dacite         |          34.9768 |        1.16954  |             75.4203 |            4.79399 |    1000 |
| 26.Black dacite             |          35.0826 |        1.73093  |             69.5072 |           10.1006  |    1000 |
| 27.Rhyolite                 |          46.2403 |        2.39781  |            150.534  |            9.78069 |    1000 |
| 28.Banded iron formation    |          41.7495 |        3.33689  |             74.6845 |            7.31814 |     999 |
| 29.Hematite breccia         |          36.9966 |        2.39865  |             63.5079 |           12.9806  |    1000 |
| 3.Gray-green sandstone      |          34.4738 |        1.61937  |            106.266  |           10.2602  |    1000 |
| 30.Altered breccia          |          40.5133 |        3.34696  |            114.151  |           12.4856  |    1000 |
| 31.Schist                   |          31.5879 |        1.18595  |             84.6889 |            5.90348 |    1000 |
| 32.Gneiss                   |          44.3418 |        2.78183  |            105.976  |            8.72421 |    1000 |
| 33.Porphyry                 |          25.46   |        1.87898  |            158.507  |           24.3513  |    1000 |
| 34.Brecciated mudstone      |          29.3515 |        5.54003  |             97.4079 |           10.1397  |    1000 |
| 35.Shale fragments          |          33.0683 |        1.935    |             95.4863 |            9.49558 |    1000 |
| 4.Gray siltstone            |          38.0173 |        3.51935  |            120.399  |           12.1799  |    1000 |
| 5.Brown siltstone           |          37.1187 |        2.50103  |            102.845  |           13.7383  |    1000 |
| 6.Carbonaceous siltstone    |          20.313  |        0.798232 |             67.9175 |            4.78379 |    1000 |
| 7.Mudstone                  |          24.407  |        5.33913  |            100.054  |           10.3587  |    1000 |
| 8.Granite                   |          44.228  |        2.20816  |            157.686  |           14.3868  |    1000 |
| 9.Megacrystic granite       |          47.7091 |        2.3831   |             87.4059 |           16.35    |    1000 |