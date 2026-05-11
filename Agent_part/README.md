# Agent Part

`Agent_part` contains the navigation-side agent services for UrbanNav.

## RouteServer

The active backend is:

- [RouteServer](RouteServer/)

RouteServer provides:

- map provider access
- natural-language destination recommendation
- itinerary decomposition
- persistent route sessions
- dynamic via-point updates
- RouteContext / LocalBEVContext
- FPV image understanding
- zero-shot visual prompt generation

Start with:

- [RouteServer README](RouteServer/README.md)

For the current main workflows, use these RouteServer examples:

- `examples/image_understanding_client.py`
- `examples/route_context_debug_client.py`
- `examples/zero_shot_visual_prompt_agent_client.py`
- `examples/route_session_client.py`
- `examples/realtime_instruction_session_client.py`
- `examples/itinerary_client.py`
- `examples/macos_prompt_recommendation_client.py`

## 目前的visual prompt pipeline运行指令

```bash
python tools/run_sidewalk_images_visual_prompt_pipeline.py \
  --images-dir "$WORK_DIR/images" \
  --trajectory-json "$WORK_DIR/trajectory.json" \
  --output-dir "$WORK_DIR/output_all" \
  --sequence-name pics_gps_batch_all \
  --camera-view camera_front \
  --sam3-output-base-dir "$WORK_DIR/sam3" \
  --run-sam3 \
  --sam3-env-name sam3 \
  --sam3-inference-script /home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/batch_sam3_demo_260124.py \
  --checkpoint-path /media/ubuntu/data1/checkpoints/checkpoint_35_merged.pt \
  --gpus 0 \
  --min-score 0.3 \
  --enable-segmentation-overlay-input \
  --enforce-alignment \
  --mode foot
  ```