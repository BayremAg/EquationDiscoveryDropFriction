import logging



class CustomFormatter(logging.Formatter):

    grey = "\x1b[37;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s "

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, rec_radord):
        log_fmt = self.FORMATS.get(rec_radord.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(rec_radord)


def get_log_obj(args, name="AlphaZeroEquation"):
    logger = logging.getLogger(name)
    logger.setLevel(args.logging_level)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(args.logging_level)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)
    logger.propagate = False
    return logger


def add_line_breaks(text, max_length):
    words = text.split()
    current_line = ""
    lines = []

    for word in words:
        if len(current_line) + len(word) + 1 > max_length:
            lines.append(current_line.strip())
            current_line = word + " "
        else:
            current_line += word + " "

    if current_line:
        lines.append(current_line.strip())

    return "\n".join(lines)


def gfk(dictionary):
    return list(dictionary.keys())[0]
