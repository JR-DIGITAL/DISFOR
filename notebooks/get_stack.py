from pathlib import Path

import pystac_client
from odc.stac import stac_load
from odc.geo.geobox import GeoBox

from pyproj import Transformer
import shapely

STAC_API = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = [
    "blue",
    "green",
    "red",
    "nir",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir08",
    "swir16",
    "swir22",
    "scl",
]

bands_mapping = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B08",
    "rededge1": "B05",
    "rededge2": "B06",
    "rededge3": "B07",
    "nir08": "B8A",
    "swir16": "B11",
    "swir22": "B12",
    "scl": "SCL",
}

desired_band_order = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
    "SCL",
]


def get_stack(sample):
    output_dir = "../data/evoland"
    output_path = Path(output_dir, f"{sample.id}")

    if output_path.exists():
        return True

    centroid = sample.geometry

    lat, lon = centroid.y, centroid.x

    start = "2015-01-01"
    end = "2025-01-01"

    # Search the catalogue
    catalog = pystac_client.Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start}/{end}",
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        query={"eo:cloud_cover": {"lt": 80}},
    )

    all_items = list(search.items())

    # Reduce to one per date (there might be some duplicates
    # based on the location)
    items = []
    dates = []
    for item in all_items:
        if item.datetime.date() not in dates:
            items.append(item)
            dates.append(item.datetime.date())

    # Extract coordinate system from first item
    epsg = items[0].properties["proj:code"]

    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    coords = shapely.transform([centroid], transformer.transform, interleaved=False)[
        0
    ].coords[0]

    # Create bounds in projection
    size = 32
    gsd = 10
    bounds = (
        coords[0] - (size * gsd) // 2,
        coords[1] - (size * gsd) // 2,
        coords[0] + (size * gsd) // 2,
        coords[1] + (size * gsd) // 2,
    )

    gbox = GeoBox.from_bbox(bounds, crs=epsg, resolution=10, tol=0.5)

    ds = (
        stac_load(
            items,
            bands=BANDS,
            groupby="solar_day",
            chunks={"time": 1},
            geobox=gbox,
            fail_on_error=False,
        )
        .compute()
        .rename(bands_mapping)
    )

    resorted = ds[desired_band_order].drop_duplicates("time")
    for time in resorted.time:
        time_str = time.astype("datetime64[us]").item().strftime("%Y-%m-%d")
        out_file = output_path / f"{time_str}.tif"
        out_file.parent.mkdir(exist_ok=True, parents=True)
        resorted.sel(time=time).rio.to_raster(out_file, driver="COG")

    return True
