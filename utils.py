import csv


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    
    # print("Pred: ")
    # print(pred)
    
    # print("Targ: ")
    # print(targets.view(1, -1))
    
    correct = pred.eq(targets.view(1, -1))
    
    print(correct)
    print(correct.float())
    print(correct.float().sum())
    print(correct.float().sum().data)
    # print(correct.float().sum().data[0])
    try:
        n_correct_elems = correct.float().sum().data[0]
    except:
        n_correct_elems = correct.float().sum().data

    return n_correct_elems / batch_size


def calculate_accuracy_single_target(outputs, targets):
    batch_size = targets.size(0)

    # print("OP: ")
    # print(outputs)
    # print("TG: ")
    # print(targets)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    
    # print("Pred: ")
    # print(pred)
    
    # print("Targ: ")
    # print(targets.view(1, -1))
    
    OP_digi = outputs > 0.5
    # print("OP digi: ")
    # print(OP_digi)
    TG_digi = targets > 0.5
    # print("TG digi: ")
    # print(TG_digi)
    
    correct = sum(OP_digi == TG_digi)
    
    # correct = pred.eq(targets.view(1, -1))
    
    # print(correct)
    # print(correct.float())
    # print(correct.float().sum())
    # print(correct.float().sum().data)
    # print(correct.float().sum().data[0])
    try:
        n_correct_elems = correct.float().sum().data[0]
    except:
        n_correct_elems = correct.float().sum().data

    return n_correct_elems / batch_size