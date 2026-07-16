
from datetime import datetime


def build_summary(
    model_name, checkpoint_path, start_epoch,
    test_ct_mr_root, img_size, prompt_type,
    test_results, num_slices, dataset_name="smc",
):
    r3 = lambda x: round(float(x), 3)
    r2 = lambda x: round(float(x), 2)

    dataset_name = str(dataset_name).lower()

    out = {
        "experiment_name": model_name,
        "checkpoint": checkpoint_path,
        "epoch": start_epoch,
        "test_data_path": test_ct_mr_root,
        "image_size": img_size,
        "prompt_type": prompt_type,
        "dataset_name": dataset_name,
        "metrics": {
            "overall": {
                "total_loss": r3(test_results["loss"]),
                "iou": {"mean": r3(test_results["iou"]), "std": r3(test_results["iou_std"])},
                "dice": {"mean": r3(test_results["dice"]), "std": r3(test_results["dice_std"])},
                "hd95": {"mean": r2(test_results["hd95"]), "std": r2(test_results["hd95_std"])},
            },
            "lesion": {
                "iou": {"mean": r3(test_results.get("iou_lesion", 0)), "std": r3(test_results.get("iou_std_lesion", 0))},
                "dice": {"mean": r3(test_results.get("dice_lesion", 0)), "std": r3(test_results.get("dice_std_lesion", 0))},
                "hd95": {"mean": r2(test_results.get("hd95_lesion", 0)), "std": r2(test_results.get("hd95_std_lesion", 0))},
                "num_samples": int(test_results.get("num_samples_lesion", 0)),
            },
        },
        "num_slices": num_slices,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if dataset_name != "apis":
        out["metrics"]["liver"] = {
            "iou": {"mean": r3(test_results.get("iou_liver", 0)), "std": r3(test_results.get("iou_std_liver", 0))},
            "dice": {"mean": r3(test_results.get("dice_liver", 0)), "std": r3(test_results.get("dice_std_liver", 0))},
            "hd95": {"mean": r2(test_results.get("hd95_liver", 0)), "std": r2(test_results.get("hd95_std_liver", 0))},
            "num_samples": int(test_results.get("num_samples_liver", 0)),
        }

    if dataset_name == "hecktor":
        out["metrics"]["tumor"] = {
            "iou":  {"mean": r3(test_results.get("iou_tumor",  0)), "std": r3(test_results.get("iou_std_tumor",  0))},
            "dice": {"mean": r3(test_results.get("dice_tumor", 0)), "std": r3(test_results.get("dice_std_tumor", 0))},
            "hd95": {"mean": r2(test_results.get("hd95_tumor", 0)), "std": r2(test_results.get("hd95_std_tumor", 0))},
            "num_samples": int(test_results.get("num_samples_tumor", 0)),
        }

    return out

# from datetime import datetime

# def build_summary(model_name, checkpoint_path, start_epoch, test_ct_mr_root, img_size, prompt_type, test_results, num_slices):
#     r3 = lambda x: round(float(x), 3)
#     r2 = lambda x: round(float(x), 2)
#     return {
#         "experiment_name": model_name,
#         "checkpoint": checkpoint_path,
#         "epoch": start_epoch,
#         "test_data_path": test_ct_mr_root,
#         "image_size": img_size,
#         "prompt_type": prompt_type,
#         "metrics": {
#             "overall": {
#                 "total_loss": r3(test_results["loss"]),
#                 "iou": {"mean": r3(test_results["iou"]), "std": r3(test_results["iou_std"])},
#                 "dice": {"mean": r3(test_results["dice"]), "std": r3(test_results["dice_std"])},
#                 "hd95": {"mean": r2(test_results["hd95"]), "std": r2(test_results["hd95_std"])},
#             },
#             "liver": {
#                 "iou": {"mean": r3(test_results.get("iou_liver", 0)), "std": r3(test_results.get("iou_std_liver", 0))},
#                 "dice": {"mean": r3(test_results.get("dice_liver", 0)), "std": r3(test_results.get("dice_std_liver", 0))},
#                 "hd95": {"mean": r2(test_results.get("hd95_liver", 0)), "std": r2(test_results.get("hd95_std_liver", 0))},
#                 "num_samples": int(test_results.get("num_samples_liver", 0)),
#             },
#             "lesion": {
#                 "iou": {"mean": r3(test_results.get("iou_lesion", 0)), "std": r3(test_results.get("iou_std_lesion", 0))},
#                 "dice": {"mean": r3(test_results.get("dice_lesion", 0)), "std": r3(test_results.get("dice_std_lesion", 0))},
#                 "hd95": {"mean": r2(test_results.get("hd95_lesion", 0)), "std": r2(test_results.get("hd95_std_lesion", 0))},
#                 "num_samples": int(test_results.get("num_samples_lesion", 0)),
#             },
#         },
#         "num_slices": num_slices,
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     }

# from datetime import datetime

# def build_summary(model_name, checkpoint_path, start_epoch, test_ct_mr_root, img_size, prompt_type, test_results, num_slices):
#     r3 = lambda x: round(float(x), 3)
#     r2 = lambda x: round(float(x), 2)
#     return {
#         "experiment_name": model_name,
#         "checkpoint": checkpoint_path,
#         "epoch": start_epoch,
#         "test_data_path": test_ct_mr_root,
#         "image_size": img_size,
#         "prompt_type": prompt_type,
#         "metrics": {
#             "overall": {
#                 "total_loss": r3(test_results["loss"]),
#                 "iou": {"mean": r3(test_results["iou"]), "std": r3(test_results["iou_std"])},
#                 "dice": {"mean": r3(test_results["dice"]), "std": r3(test_results["dice_std"])},
#                 "hd95": {"mean": r2(test_results["hd95"]), "std": r2(test_results["hd95_std"])},
#             },
#             "liver": {
#                 "iou": {"mean": r3(test_results.get("iou_liver", 0)), "std": r3(test_results.get("iou_std_liver", 0))},
#                 "dice": {"mean": r3(test_results.get("dice_liver", 0)), "std": r3(test_results.get("dice_std_liver", 0))},
#                 "hd95": {"mean": r2(test_results.get("hd95_liver", 0)), "std": r2(test_results.get("hd95_std_liver", 0))},
#                 "num_samples": int(test_results.get("num_samples_liver", 0)),
#             },
#             "lesion": {
#                 "iou": {"mean": r3(test_results.get("iou_lesion", 0)), "std": r3(test_results.get("iou_std_lesion", 0))},
#                 "dice": {"mean": r3(test_results.get("dice_lesion", 0)), "std": r3(test_results.get("dice_std_lesion", 0))},
#                 "hd95": {"mean": r2(test_results.get("hd95_lesion", 0)), "std": r2(test_results.get("hd95_std_lesion", 0))},
#                 "num_samples": int(test_results.get("num_samples_lesion", 0)),
#             },
#              "case_level": {
#                 "overall": {
#                     "dice": {"mean": r3(test_results.get("case_dice", 0))},
#                     "iou": {"mean": r3(test_results.get("case_iou", 0))},
#                     "hd95": {"mean": r2(test_results.get("case_hd95", 0))},
#                 },
#                 "liver": {
#                     "dice": {"mean": r3(test_results.get("case_dice_liver", 0)), "std": r3(test_results.get("case_dice_std_liver", 0))},
#                     "iou": {"mean": r3(test_results.get("case_iou_liver", 0)), "std": r3(test_results.get("case_iou_std_liver", 0))},
#                     "hd95": {"mean": r2(test_results.get("case_hd95_liver", 0)), "std": r2(test_results.get("case_hd95_std_liver", 0))},
#                     "num_cases": int(test_results.get("num_cases_liver", 0)),
#                 },
#                 "lesion": {
#                     "dice": {"mean": r3(test_results.get("case_dice_lesion", 0)), "std": r3(test_results.get("case_dice_std_lesion", 0))},
#                     "iou": {"mean": r3(test_results.get("case_iou_lesion", 0)), "std": r3(test_results.get("case_iou_std_lesion", 0))},
#                     "hd95": {"mean": r2(test_results.get("case_hd95_lesion", 0)), "std": r2(test_results.get("case_hd95_std_lesion", 0))},
#                     "num_cases": int(test_results.get("num_cases_lesion", 0)),
#                 },
#             },
#         },
#         "num_slices": num_slices,
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     }