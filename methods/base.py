from layers import AvgDepthErrorsPerDepthRange

class BaseAdaptation:
    def __init__(self, opt, **kwargs):
        self.opt = opt

        self.generate_images_pred = kwargs['gen_imgs_pred_func']
        self.predict_poses = kwargs['predict_poses_func']
        self.compute_losses_unsup = kwargs['compute_losses_unsup_func']
        self.adapt_instance = kwargs['adapt_instance']
        
        self.avg_depth_errors_median_scaling = AvgDepthErrorsPerDepthRange(self.opt, median_scaling=True)
        self.avg_depth_errors = AvgDepthErrorsPerDepthRange(self.opt, median_scaling=False)

    def process_batch(self, inputs):
        raise NotImplementedError("Subclasses must implement this method")
