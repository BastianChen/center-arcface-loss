from train import Trainer

if __name__ == '__main__':
    # trainer = Trainer(r"models/conv_center_arcface.pth", r"images/")
    # trainer = Trainer(r"models/conv_arcface.pth", r"images_arcface/")
    trainer = Trainer(r"models/conv_center.pth", r"images_center/")
    trainer.train()
