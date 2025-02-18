import torch
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2
from PIL import Image
import os
import gradio as gr

class MaskGenerator:
    def __init__(self, sam_checkpoint):
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam.to("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SamPredictor(self.sam)
        
    def generate_mask(self, image, x, y, points):
        if isinstance(image, dict):
            image = image['image']
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        self.predictor.set_image(image)
        
        # Add new point if coordinates are provided
        if x is not None and y is not None:
            points.append([x, y])
            
        if not points:
            return image, points, None
            
        # Create visualization
        vis_image = image.copy()
        
        # Draw all points
        for px, py in points:
            cv2.circle(vis_image, (int(px), int(py)), 5, (255, 0, 0), -1)
        
        # Generate mask
        input_points = np.array(points)
        input_labels = np.array([1] * len(points))
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        # Select mask with highest score
        mask = masks[np.argmax(scores)]
        
        # Apply mask overlay
        vis_image[mask > 0] = vis_image[mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
        
        return vis_image, points, mask

def create_mask_app(image_dir, mask_dir, sam_checkpoint):
    mask_generator = MaskGenerator(sam_checkpoint)
    
    def add_point(image, x, y, points):
        if points is None:
            points = []
        return mask_generator.generate_mask(image, x, y, points)

    def clear_points(image):
        return image, [], None
        
    def save_mask(image_name, mask, is_cat):
        if mask is None:
            return "No mask to save"
            
        base_name = os.path.splitext(image_name)[0]
        suffix = "_cat.png" if is_cat else "_dog.png"
        mask_path = os.path.join(mask_dir, f"{base_name}{suffix}")
        cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
        return f"Saved mask to {mask_path}"

    with gr.Blocks() as app:
        gr.Markdown("""
        ## Interactive Mask Generator
        1. Enter X and Y coordinates to add points marking the object
        2. The green overlay shows the current mask
        3. Click 'Clear Points' to start over
        4. Save as either cat or dog mask when satisfied
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Image", type="numpy")
                with gr.Row():
                    x_coord = gr.Number(label="X coordinate", precision=0)
                    y_coord = gr.Number(label="Y coordinate", precision=0)
                add_point_btn = gr.Button("Add Point")
                points_state = gr.State([])
                mask_state = gr.State(None)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Points")
                    save_cat_btn = gr.Button("Save as Cat Mask")
                    save_dog_btn = gr.Button("Save as Dog Mask")
                    
            mask_output = gr.Image(label="Generated Mask", type="numpy")
            
        save_output = gr.Textbox(label="Save Status")
        
        # Event handlers
        add_point_btn.click(
            fn=add_point,
            inputs=[image_input, x_coord, y_coord, points_state],
            outputs=[image_input, points_state, mask_state]
        )
        
        clear_btn.click(
            fn=clear_points,
            inputs=[image_input],
            outputs=[image_input, points_state, mask_state]
        )
        
        save_cat_btn.click(
            fn=save_mask,
            inputs=[gr.State(os.path.basename(image_dir)), mask_state, gr.State(True)],
            outputs=save_output
        )
        
        save_dog_btn.click(
            fn=save_mask,
            inputs=[gr.State(os.path.basename(image_dir)), mask_state, gr.State(False)],
            outputs=save_output
        )
        
    return app

def main():
    image_dir = "datasets/my_data/characters/real/cat2_dog6/image"
    mask_dir = "datasets/my_data/characters/real/cat2_dog6/mask"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    
    os.makedirs(mask_dir, exist_ok=True)
    
    app = create_mask_app(image_dir, mask_dir, sam_checkpoint)
    app.launch(share=True)  # share=True creates a public URL

if __name__ == "__main__":
    main()