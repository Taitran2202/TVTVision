def performances(gt_cnts, pred_cnts):
    gt_cnts = gt_cnts.cpu()
    pred_cnts = pred_cnts.cpu()
    val_mae = abs(pred_cnts - gt_cnts)
    val_rmse = (pred_cnts - gt_cnts) ** 2
    return val_mae, val_rmse
