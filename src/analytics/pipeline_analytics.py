# %%

from exec_query import exec_query

import datetime

now = datetime.datetime.now().strftime("%Y-%m-%d")

steps = [
    {
        "table":"life_cycle",
        "db_origin":"loyalty-system",
        "db_target":"analytics",
        "dt_start":now,
        "dt_stop":now,
        "monthly":False,
        "mode":"append",
    },
    {
        "table":"fs_transacional",
        "db_origin":"loyalty-system",
        "db_target":"analytics",
        "dt_start":now,
        "dt_stop":now,
        "monthly":False,
        "mode":"append",
    },
    {
        "table":"fs_education",
        "db_origin":"education-platform",
        "db_target":"analytics",
        "dt_start":now,
        "dt_stop":now,
        "monthly":False,
        "mode":"append",
    },
    {
        "table":"fs_life_cycle",
        "db_origin":"analytics",
        "db_target":"analytics",
        "dt_start":now,
        "dt_stop":now,
        "monthly":False,
        "mode":"append",
    },
    {
        "table":"fs_all",
        "db_origin":"analytics",
        "db_target":"analytics",
        "dt_start":now,
        "dt_stop":now,
        "monthly":False,
        "mode":"replace",
    },
]

for s in steps:
    exec_query(**s)
