import cv2
import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics import YOLO

class YoloPredictor():
    def __init__(self, model_name): 
        self.model_name = model_name      # The detection model name to use
        self.model = YOLO(model_name)       # the model
        self.conf = 0.8             # threshold of confidence level
        self.iou = 0.45              # intersection of union threshold
    # drawing detection box
    def draw_boxes(self, image, results, tags):
        """
        draw the detection box of image.

        Args:
            image (numpy.array()): the numpy array storing the image data prepared to be drew box.
            results (list of results objects): results containing label information.
            
        Returns:
            img (numpy.array()): the pocessed numpy array staring the boxed image.
        """
        for result in results:
            boxes = result.boxes
            for box in boxes:
                label = self.model.names[int(box.cls[0])]
                if len(tags) !=0 and label not in tags:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].numpy()
                
                confidence = box.conf[0].numpy()
                
                # Draw the bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Put the label and confidence
                if (int(y1) < 20):
                    cv2.putText(image, f'{label} {confidence:.2f}', (int(x1) + 3, int(y1) + 22), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(image, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
        return image
    # main for detect
    def inference(self, img, tags):
        """
        infer on the image using yolov8 model.

        Args:
            img (numpy.array()): the numpy array storing the image data prepared to be processed.

        Returns:
            labels_count_dict (dictionary): detected labels with the corresponding detection number.
            frame (numpy.array()): the pocessed numpy array image with detection box.
            
        """
        if not self.model:
            self.setup_model(self.new_model_name)
        # start detection

        # inference 
        results = self.model(img, conf = self.conf, iou = self.iou)
        
        frame = self.draw_boxes(img, results, tags)
        class_names = self.model.names
        labels_count_dict = {}
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                label = class_names[class_id]
                if len(tags) !=0 and label not in tags:
                    continue
                if label not in labels_count_dict:
                    labels_count_dict[label] = 0
                
                labels_count_dict[label] += 1

        return labels_count_dict, frame
    
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        """
        Convert image variable from numpy array to torch tensor and rescale the tensor.

        Args:
            img (numpy.array()): the numpy array storing the image data prepared to be preprocessed.

        Returns:
            img (torch.tensor()): the prerpocessed tensor storing the image data
        """
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        """
        Adding restriction to the prediction result and wrap the prediction results to Results class.

        Args:
            preds (torch.tensor()): The prediction result as tensor.
            img (torch.tensor()): The original image as tensor.
            orig_img (numpy.array()): the original image as numpy array.

        Returns:
            results (Results): the Results object wrapping prediction results.
        """
        ### important
        preds = ops.non_max_suppression(preds,
                                        self.conf_thres,
                                        self.iou_thres,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1         # attention
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        self.annotator = self.get_annotator(im0)
        det = results[idx].boxes

        if len(det) == 0:
            return f'{log_string}(no detections), ' 

        for c in det.cls.unique():
            n = (det.cls == c).sum()
            log_string += f"{n}~{self.model.names[int(c)]}," 


        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.save_txt:  # Write to file
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))
                
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.save_res or self.args.save_crop or self.args.show or True:
                c = int(cls)  # integer class
                name = f'id:{int(d.id.item())} {self.model.names[c]}' if d.id is not None else self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop: # save cropped image
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string