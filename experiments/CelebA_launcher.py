from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            dataroot="D:\projects\CelebAMask-HQ\CelebAMask-HQ\CelebA-HQ-img",
            dataroot2="D:\projects\CelebAMask-HQ\CelebAMask-HQ\CelebA-HQ-img",
            dataset_mode="CelebAMask",
            checkpoints_dir="./checkpoints/",
            num_gpus=1, batch_size=2,
            preprocess="resize",
            load_size=512, crop_size=512,
        )

        return [
            opt.specify(
                name="CelebA_train",
                model="DGPST",
                optimizer="Diff",
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            continue_train=True,
            evaluation_metrics="swap_visualization",
            evaluation_freq=50000) for opt in common_options]
        
    def test_options(self):
        opts = self.options()[0]
        return [
            opts.tag("swapping_grid").specify(
                num_gpus=1,
                batch_size=1,
                name="CelebA_default",
                dataroot="/path/to/your/image/folders/",
                dataname = "your experiment name",
                dataset_mode="CelebAMask",
                preprocess="scale_width",
                evaluation_metrics="structure_style_grid_generation"
            ),
        ]
