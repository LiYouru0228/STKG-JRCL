# Upload Data Guidelines:
Upload the dataset for your own task here. The format of the dataset should be organized as a python 'dict', and it contains the following parts:
+ Part 1:
    + key:"entityid2baseinfo"
    + value:
        + type: dict
        + format: {entityid: one-hot encoding baseinfo list}
+ Part 2:
    + key:"entityid2lbs_info"
    + value:
        + type: dict
        + format: {entityid: lbs info}
+ Part 3:
    + key:"dkg"
    + value:
        + type: dict
        + format: {'time_interval': {head entityId: [(tail entityId, relationId),...]},...}
+ Part 4:
    + key:"entityid2type"
    + value:
        + type: dict
        + format: {entityId: "B"/"C"/"I","P", etc.}
+ Part 5:
    + key: "sample_dict"
    + value:
        + type: dict
        + format: {dataset name: numpy.ndarray}
           +  raw demo: <enterpriseID1, enterpriseID2, lable>
