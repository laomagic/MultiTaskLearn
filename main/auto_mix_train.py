
# 混合精度的训练 o1 是混合精度 o0是fp32 o2几乎FP16  o3纯FP16

# from apex import amp
# model, optimizer = amp.initialize(model, optimizer, opt_level="O1") 
# with amp.scale_loss(loss, optimizer) as scaled_loss:
#     scaled_loss.backward()