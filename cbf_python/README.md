# cbf_python Readme

OPTUNA: if you want to install the postrgresql db on a docker, run:

```bash
docker run --name optuna-postgres \
  -e POSTGRES_USER=optuna \
  -e POSTGRES_PASSWORD=optuna_pw \
  -e POSTGRES_DB=optuna_db \
  -v optuna_postgres_data:/var/lib/postgresql \
  -p 5432:5432 \
  -d postgres
```
to run the container after a restart:

```bash
docker start optuna-postgres
```