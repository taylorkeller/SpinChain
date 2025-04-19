import os
import sys
import yaml
import webbrowser

def validate_paths(opt):
    print("📋 Checking paths...")
    assert os.path.exists(opt['weights']), f"❌ weights not found: {opt['weights']}"
    assert os.path.exists(opt['data']), f"❌ data.yaml not found: {opt['data']}"
    assert os.path.exists("val.py"), "❌ YOLOv5 val.py not found!"
    print("✅ Paths OK")

def load_data_yaml(data_path):
    with open(data_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    train_path = data_cfg.get('train', '')
    val_path = data_cfg.get('val', '')
    return train_path, val_path

def check_train_val_overlap(train_path, val_path):
    if train_path == val_path:
        print("⚠️ WARNING: train and val paths are the same! You're evaluating on training data.")
    else:
        print("✅ Detected separate train and val sets")

def simulate_cli_args(opt):
    sys.argv = [
        "val.py",
        "--data", opt['data'],
        "--weights", opt['weights'],
        "--img", str(opt['img']),
        "--conf", str(opt['conf']),
        "--iou", str(opt['iou']),
        "--batch-size", str(opt['batch_size']),
        "--task", "val",
        "--project", opt['project'],
        "--name", opt['name'],
        "--exist-ok",
        "--save-txt",
        "--save-conf"
    ]

def run_validation(opt):
    print("🚀 Starting YOLOv5 validation...")
    import val
    parsed_opt = val.parse_opt()
    result = val.main(parsed_opt)
    if result and hasattr(result, 'plot_cm'):
        print("📊 Saving confusion matrix...")
        result.plot_cm(save_dir=parsed_opt.save_dir)
        cm_path = os.path.join(parsed_opt.save_dir, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            print("📂 Confusion matrix saved. Opening...")
            webbrowser.open(cm_path)
    else:
        print("⚠️ Could not generate confusion matrix (possibly older YOLOv5 version).")

def main():
    # Customize below:
    opt = {
        "weights": os.path.abspath("runs/train/spinchain-yolo2/weights/best.pt"),
        "data": os.path.abspath("../beyblade_dataset/data.yaml"),
        "img": 640,
        "conf": 0.001,
        "iou": 0.6,
        "batch_size": 32,
        "project": "eval_output",
        "name": "spinchain_eval"
    }

    print(f"🔧 Using interpreter: {sys.executable}")
    sys.path.append(os.getcwd())

    validate_paths(opt)

    # Step 1: check data.yaml
    train_path, val_path = load_data_yaml(opt['data'])
    print(f"📁 train: {train_path}")
    print(f"📁 val:   {val_path}")
    check_train_val_overlap(train_path, val_path)

    # Step 2: Run validation
    simulate_cli_args(opt)
    run_validation(opt)

if __name__ == "__main__":
    main()
