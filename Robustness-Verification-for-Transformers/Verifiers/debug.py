def print_tensors_info(variables):
    for name, var in variables.items():
        try:
            dev = var.device
            print("%s - device '%s'" % (name, dev))
        except AttributeError:
            pass
