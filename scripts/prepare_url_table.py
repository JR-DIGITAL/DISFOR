from utils import polars_url_table

if __name__ == "__main__":
    base_url = "https://label.nadir.earth/?"
    url_table = polars_url_table(base_url)
    url_table.write_parquet("docs/usage/exploration-table.pq")
