# Upload Data Guidelines:
Upload the dataset for your own task here. The format of the dataset should be organized as a python 'dict', and it contains the following parts:
+ Part 1:
    + key:"entityid2baseinfo"
    + value:
        + type: dict
        + format: {entityid: one-hot encoding baseinfo list}
+ Part 2:
    + key:"kg"
    + value:
        + type: dict
        + format: {head entityId: [(tail entityId, relationId),...]}
+ Part 3:
    + key:"entityid2type"
    + value:
        + type: dict
        + format: {entityId: "B"/"C", etc.}
+ Part 4:
    + key: "sample_dict"
    + value:
        + type: dict
        + format: {dataset name: numpy.ndarray}
           +  raw demo: <enterpriseID1, enterpriseID2, lable>
