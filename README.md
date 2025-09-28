# LEON: A New Framework for ML-Aided Query Optimization
---

**News**

* 🎉  LEON has been accepted to VLDB 2023. Meanwhile, the LEON project undergoes notable enhancements. For detailed information, please visit the following link: [LeonProject](https://github.com/Thisislegit/LeonProject). Subsequent updates will also be conducted through this repository.
---

Pytorch implementation of LEON: A New Framework for ML-Aided Query Optimization.

##  Requirment  

### Python Environment

```
python 3.8.5
pandas
torch==1.4.0
psycopg2-binary==2.8.5
numpy==1.18.1
networkx
ipdb
Pillow==9.2.0
scikit-learn==1.0.2
scipy==1.7.3
sqlparse
re
```

Run `pip install -r requirements.txt`  to quickly install Python Environment.

### PostgreSQL 

Postgres v12.5

pg_hint_plan v1.3.7

*After installing PostgreSQL and its extension, you need to modify its default configuration that can find in [postgresql.conf](./postgresql.conf)*

### BenchMark

In our paper,we use two benchmark, JOB and TPC-H, you can get it through the following link.

**Join-order-benchmark:** 	https://github.com/gregrahn/join-order-benchmark

**TPC-H:**  https://github.com/electrum/tpch-dbgen

## Usage

First, you need to modify the log_path, model_path and other parameters in the training code(such as [train_job.py](./train_job.py)), and modify the relevant information required to connect to PostgreSQL in [pg_executor.py](./util/pg_executor.py).

For ML-aided query optimizer, we have two implementations. One is based on PostgreSQL itself, which needs to modify its source code; The second is to use Python to simulate the dynamic programming of PG.

For the former, you need to use [allpaths.c](./allpaths.c) to replace the corresponding source code file with the same name of PG, and recompile it. The path of the file is **/src/backend/optimizer/path/allpaths.c** 

Modify the relevant path in the code and start the modified PostgreSQL,Run the following command:

```
python3 [-u] pg_train.py [ > runninglog_path/log/txt 2>&1 ]
```

For the second search mode, you can run the following command:

```
python3 [-u] train_Job.py [> runninglog_path/log.txt 2>&1 ]
or 
python3 [-u] train_tpch.py [> runninglog_path/log.txt 2>&1 ]
```



## Contact

If you have any questions about the code, please email [XUCHEN.2019@outlook.com](mailto:XUCHEN.2019@outlook.com), [HaiTian_Chen@outlook.com](mailto:HaiTian_Chen@outlook.com)
#######################
1. Tải và cài PgSQL
2. Tài và cài extension pg_hint_plan
3. Tải database (khác dataset) IMDB: The CSV files used in the paper, which are from May 2013, can be found at http://event.cwi.nl/da/job/imdb.tgz
4. Import database vào, tạo database tên là imdb, chạy schematext.sql để tạo bảng, sau đó import các file csv vào 

Giờ mới bị tiếp nè, bài báo này từ 2020 nên 1 số thư viện không còn tồn tại version cũ nữa, phải upgrade lên version mới hơn -> sửa code 
5. Tải source code LEON
6. Bật môi trường ảo venv lên 
7. Cài đặt theo như mô tả (lỗi)
8. Sửa code và sửa version các thư viện => cài đặt thành công 
9. Sửa util/pg_executor.py nhập thông tin để kết nối đến database vào 
10. Chạy lệnh python3 train_Job.py > logs/job_train.log 2>&1 để train
11. Chờ, nếu nó chạy xong nhanh tức là lỗi -> coi  logs/job_train.log để xem log lỗi, fix lỗi rồi chạy lại. Nếu lâu tức là huấn luyện được rồi, chờ thôi, tầm 30p trên macos m1

