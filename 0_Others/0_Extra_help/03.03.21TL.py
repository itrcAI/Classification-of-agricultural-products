#####er_TL
##Load##
model_TLL = PseTae(**model_args)
model_TLL.load_state_dict(
    torch.load(os.path.join('./results_we', 'Fold_{}'.format(fold + 1), 'model.pth.tar'))['state_dict'])

## Shows the weight
model_weights = model_TLL.state_dict()
first_layer_weights = model_weights['spatial_encoder.mlp1[0].lin.weight']  # Assuming first linear layer in `spatial_encoder.mlp1`
print("first_layer_weights_loadweight:", first_layer_weights)

## Shows the Model
model_tl = model
print("model_tl_Total:",model_tl)
sum_model = summary(model_tl)
print("Summary of the model_befor: ",sum_model )

### Freeze The layers  
for param in model_tl.spatial_encoder.mlp1[0].lin.parameters():
    param.requires_grad = False

#################################
###Total Main
model = PseTae(**model_args)
model.load_state_dict(
    torch.load(os.path.join('./results_we', 'Fold_{}'.format(fold + 1), 'model.pth.tar'))['state_dict'])

for param in model.spatial_encoder.mlp1[0].lin.parameters():
 param.requires_grad = False

#model.apply(weight_init)