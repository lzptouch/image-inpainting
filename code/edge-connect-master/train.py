from multiprocessing import freeze_support

from main import main
if __name__ == '__main__':
    freeze_support()
    main(mode=1)