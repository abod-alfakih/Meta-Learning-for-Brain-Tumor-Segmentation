# from .blocks import PlainBlock, ResidualBlock
# from .unet import UNet, MultiEncoderUNet
# from monai.networks.nets import SwinUNETR
# from torchsummary import summary
#
# block_dict = {
#     'plain': PlainBlock,
#     'res': ResidualBlock
# }
#
#
# def get_unet(args):
#     kwargs = {
#         "input_channels"   : args.input_channels,
#         "output_classes"   : args.num_classes,
#         "channels_list"    : args.channels_list,
#         "deep_supervision" : args.deep_supervision,
#         "ds_layer"         : args.ds_layer,
#         "kernel_size"      : args.kernel_size,
#         "dropout_prob"     : args.dropout_prob,
#         "norm_key"         : args.norm,
#         "block"            : block_dict[args.block],
#     }
#
#     if args.unet_arch == 'unet':
#         return UNet(**kwargs)
#
#     elif args.unet_arch == 'multiencoder_unet':
#         return MultiEncoderUNet(**kwargs)
#     elif args.unet_arch == 'SwinUNETR':
#         return SwinUNETR(
#             in_channels=args.input_channels,
#             out_channels=args.num_classes,
#             img_size=(args.patch_size, args.patch_size, args.patch_size),
#
#
#             feature_size=24,
#             drop_rate=args.dropout_prob,
#             norm_name=args.norm,
#             use_checkpoint=False,
#             spatial_dims=3,
#             downsample="merging",
#             use_v2=False,
#         )
#     else:
#         raise NotImplementedError(args.unet_arch + " is not implemented.")


# from .blocks import PlainBlock, ResidualBlock
# from .unet import UNet, MultiEncoderUNet
# from monai.networks.nets import ViT
# from torchsummary import summary
#
# block_dict = {
#     'plain': PlainBlock,
#     'res': ResidualBlock
# }
#
#
# def get_unet(args):
#     kwargs = {
#         "input_channels"   : args.input_channels,
#         "output_classes"   : args.num_classes,
#         "channels_list"    : args.channels_list,
#         "deep_supervision" : args.deep_supervision,
#         "ds_layer"         : args.ds_layer,
#         "kernel_size"      : args.kernel_size,
#         "dropout_prob"     : args.dropout_prob,
#         "norm_key"         : args.norm,
#         "block"            : block_dict[args.block],
#     }
#
#     if args.unet_arch == 'unet':
#         return UNet(**kwargs)
#
#     elif args.unet_arch == 'multiencoder_unet':
#         return MultiEncoderUNet(**kwargs)
#     elif args.unet_arch == 'VIT':
#         return ViT(
#             in_channels=args.input_channels,
#             img_size=(args.patch_size, args.patch_size, args.patch_size),
#             patch_size=16,
#             hidden_size=768,
#             mlp_dim=3072,
#             num_heads=12,
#             pos_embed='perceptron',
#             classification=False,
#             dropout_rate=args.dropout_prob,
#             spatial_dims=3
#         )
#     else:
#         raise NotImplementedError(args.unet_arch + " is not implemented.")

#
# from .blocks import PlainBlock, ResidualBlock
# from .unet import UNet, MultiEncoderUNet
# from monai.networks.nets import BasicUNetPlusPlus
# from torchsummary import summary
#
# block_dict = {
#     'plain': PlainBlock,
#     'res': ResidualBlock
# }
#
#
# def get_unet(args):
#     kwargs = {
#         "input_channels"   : args.input_channels,
#         "output_classes"   : args.num_classes,
#         "channels_list"    : args.channels_list,
#         "deep_supervision" : args.deep_supervision,
#         "ds_layer"         : args.ds_layer,
#         "kernel_size"      : args.kernel_size,
#         "dropout_prob"     : args.dropout_prob,
#         "norm_key"         : args.norm,
#         "block"            : block_dict[args.block],
#     }
#
#     if args.unet_arch == 'unet':
#         return UNet(**kwargs)
#
#     elif args.unet_arch == 'multiencoder_unet':
#         return MultiEncoderUNet(**kwargs)
#     elif args.unet_arch == 'BasicUNetPlusPlus':
#         return BasicUNetPlusPlus(
#             spatial_dims=3,  # Keep spatial dims as 3 for 3D images
#             in_channels=args.input_channels,
#             out_channels=args.num_classes,
#             features=args.channels_list,  # Features are passed from channels_list
#             deep_supervision=args.deep_supervision,  # Whether to use deep supervision
#             act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),  # Activation function
#             norm=("instance", {"affine": True}),  # Normalization
#             bias=True,  # Keep bias as True unless specified otherwise
#             dropout=args.dropout_prob,  # Dropout probability
#             upsample="deconv",  # Use deconvolution for upsampling
#         )
#     else:
#         raise NotImplementedError(args.unet_arch + " is not implemented.")

#


from .blocks import PlainBlock, ResidualBlock
from .unet import UNet, MultiEncoderUNet
from monai.networks.nets import UNETR
from torchsummary import summary

block_dict = {
    'plain': PlainBlock,
    'res': ResidualBlock
}


def get_unet(args):
    kwargs = {
        "input_channels": args.input_channels,
        "output_classes": args.num_classes,
        "channels_list": args.channels_list,
        "deep_supervision": args.deep_supervision,
        "ds_layer": args.ds_layer,
        "kernel_size": args.kernel_size,
        "dropout_prob": args.dropout_prob,
        "norm_key": args.norm,
        "block": block_dict[args.block],
    }

    if args.unet_arch == 'unet':
        return UNet(**kwargs)

    elif args.unet_arch == 'multiencoder_unet':
        return MultiEncoderUNet(**kwargs)
    elif args.unet_arch == 'unetr':
        return UNETR(
            spatial_dims=3,
            in_channels=args.input_channels,
            out_channels=args.num_classes,
            img_size=(args.patch_size, args.patch_size, args.patch_size),
            norm_name=args.norm,
            dropout_rate=args.dropout_prob,
        )
    else:
        raise NotImplementedError(args.unet_arch + " is not implemented.")