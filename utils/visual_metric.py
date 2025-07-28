import matplotlib.pyplot as plt
def visual_metric(log_path):
    with open(log_path) as f:
        logs = f.readlines()
    
    pre = []
    rec = []
    f1 = []
    mae = []
    
    for line in logs:
        if "real test val" in line:
            line_list = line.split()[8:]
            if len(line_list) == 3:
                pre.append(float(line_list[0].split(":")[1][:-1]))
                rec.append(float(line_list[1].split(":")[1][:-1]))
                f1.append(float(line_list[2].split(":")[1]))
            elif len(line_list) == 2:
                mae.append(float(line_list[0].split(":")[1][:-1]))
    
    assert len(pre) == len(rec) and len(rec) == len(f1) and len(f1) == len(mae)
    fig, ax = plt.subplots()
    x = [i+1 for i in range(len(pre))]
    ax.plot(x, pre, label = "pre")
    ax.plot(x, rec, label = "rec")
    ax.plot(x, f1, label = "f1")
    # ax.plot(x, mae, label = "mae")
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.set_title('Simple Plot')
    ax.legend()

    plt.savefig("test.jpg")
if __name__ == "__main__":
    visual_metric("ablation_study3/Mall/scene_001/0930-160247/train.log")

    

    
