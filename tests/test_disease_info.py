def test_every_class_has_one_chinese_name_and_description():
    from config import CLASS_IDS, CLASS_NAMES_CN
    from disease_info import DISEASE_INFO

    assert set(CLASS_NAMES_CN) == set(CLASS_IDS)
    assert set(DISEASE_INFO) == set(CLASS_IDS)

    for class_id in CLASS_IDS:
        info = DISEASE_INFO[class_id]
        assert info.display_name == CLASS_NAMES_CN[class_id]
        assert info.display_name.strip()
        assert info.description.strip()


def test_descriptions_are_symptom_only_and_non_prescriptive():
    from disease_info import DISEASE_INFO

    prohibited = ("剂量", "农药", "喷洒", "用药", "处方", "治疗方案")

    for info in DISEASE_INFO.values():
        assert not any(term in info.description for term in prohibited)
