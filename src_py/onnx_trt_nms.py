import onnx_graphsurgeon as gs
import onnx
import numpy as np

# 加载原始 ONNX 模型
onnx_path = "/home/crrcdt123/git/ultralytics/runs/detect/train24/weights/best.onnx"
output_path = "/home/crrcdt123/git/ultralytics/runs/detect/train24/weights/best_nms.onnx"
graph = gs.import_onnx(onnx.load(onnx_path))

# 获取原始输出：假设 shape 是 [1, 8， 8400]
raw_output = graph.outputs[0]
num_classes = 4

# Step 1: Transpose to [1, 8400, 8]
transposed = gs.Variable(name="output_transposed", dtype=np.float32)
transpose_node = gs.Node(op="Transpose", name="transpose_output",
                         attrs={"perm": [0, 2, 1]},
                         inputs=[raw_output],
                         outputs=[transposed])
graph.nodes.append(transpose_node)

# 拆分 boxes 和 scores
boxes = gs.Variable(name="srcboxes", dtype=np.float32)
scores = gs.Variable(name="srcscores", dtype=np.float32)

split_node = gs.Node(op="Split", name="split_boxes_scores",
                     attrs={"axis": 2, "split": [4, num_classes]},
                     inputs=[transposed],
                     outputs=[boxes, scores])
graph.nodes.append(split_node)

# 添加 EfficientNMS_TRT 插件节点
nms_num_det = gs.Variable(name="num_dets", dtype=np.int32)
nms_boxes = gs.Variable(name="boxes", dtype=np.float32)
nms_scores = gs.Variable(name="scores", dtype=np.float32)
nms_classes = gs.Variable(name="labels", dtype=np.int32)

nms_node = gs.Node(
    op="EfficientNMS_TRT",
    name="efficientnms_trt",
    inputs=[boxes, scores],
    outputs=[nms_num_det, nms_boxes, nms_scores, nms_classes],
    attrs={
        "plugin_version": "1",
        "background_class": -1,
        "max_output_boxes": 100,
        "score_threshold": 0.25,
        "iou_threshold": 0.45,
        "score_activation": False,
        "box_coding": 1,  # 0: [x1, y1, x2, y2]
    }
)
graph.nodes.append(nms_node)

# 设置新的输出
graph.outputs = [nms_num_det, nms_boxes, nms_scores, nms_classes]

# 清理图并导出新模型
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), output_path)

print(f"✅ 新模型已保存: {output_path}")
