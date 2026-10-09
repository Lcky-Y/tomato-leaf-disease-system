"""Chinese display metadata for the fixed tomato disease labels."""

from dataclasses import dataclass

from config import CLASS_NAMES_CN


@dataclass(frozen=True)
class DiseaseInfo:
    display_name: str
    description: str


DISEASE_INFO: dict[str, DiseaseInfo] = {
    "Tomato___Bacterial_spot": DiseaseInfo(
        CLASS_NAMES_CN["Tomato___Bacterial_spot"],
        "叶片常见深褐色小斑点，周围可能出现黄色晕圈。",
    ),
    "Tomato___Early_blight": DiseaseInfo(
        CLASS_NAMES_CN["Tomato___Early_blight"],
        "叶面常见带同心轮纹的褐色斑块，老叶症状通常更明显。",
    ),
    "Tomato___Late_blight": DiseaseInfo(
        CLASS_NAMES_CN["Tomato___Late_blight"],
        "叶片可能出现不规则的水浸状深色病斑，并逐渐扩大。",
    ),
    "Tomato___Leaf_Mold": DiseaseInfo(
        CLASS_NAMES_CN["Tomato___Leaf_Mold"],
        "叶面可见淡黄色斑区，叶背对应区域可能呈橄榄色霉层。",
    ),
    "Tomato___Septoria_leaf_spot": DiseaseInfo(
        CLASS_NAMES_CN["Tomato___Septoria_leaf_spot"],
        "叶片常出现数量较多的圆形小斑，中心偏灰、边缘颜色较深。",
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": DiseaseInfo(
        CLASS_NAMES_CN["Tomato___Spider_mites Two-spotted_spider_mite"],
        "叶面可能出现密集浅色失绿点，严重时叶片发黄或干枯。",
    ),
    "Tomato___Target_Spot": DiseaseInfo(
        CLASS_NAMES_CN["Tomato___Target_Spot"],
        "叶片可见褐色圆斑，部分病斑具有较明显的环状纹理。",
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": DiseaseInfo(
        CLASS_NAMES_CN["Tomato___Tomato_Yellow_Leaf_Curl_Virus"],
        "叶片可能黄化、边缘上卷，叶片整体变小并伴随皱缩。",
    ),
    "Tomato___Tomato_mosaic_virus": DiseaseInfo(
        CLASS_NAMES_CN["Tomato___Tomato_mosaic_virus"],
        "叶片常呈深浅绿色相间的花叶状纹理，并可能发生变形。",
    ),
    "Tomato___healthy": DiseaseInfo(
        CLASS_NAMES_CN["Tomato___healthy"],
        "叶片颜色和纹理较均匀，未见训练类别中的典型病斑特征。",
    ),
}
