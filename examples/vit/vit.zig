const std= @import("std");
const zml= @import("zml");
const stdx= zml.stdx;

const log= std.log.scoped(.vit);

// specifying the types of operations that can be done in a ViT
pub const Pooling= enum{cls, mean, none};

// ViT config based off Hugging face transformers repo
pub const VitConfig= struct {
patch_size:u32,
image_size: stdx.json.Union(union(enum) {
      int: u32,
      ints:[]u32
}),
layer_norm_eps:f32=1e-6,
input_num_channels:u32=3,
num_attention_heads:u32,
hidden_size: u32,
mlp_size:u32,
num_hidden_layers:u32,
qkv_bias: bool=true,
hidden_act: zml.nn.Activation= .gelu,
pooler_act: zml.nn.Activation =.tanh,
pooler_output_size: u32,

// getting image size as a 2d dimension

pub fn imageWh(self: VitConfig) [2]u32 {
    return switch(self.image_size.value){
        .int => |v| .{v,v},
        .ints => |s| .{s[0], s[1]}
    };
}

// getting patches as a 2D 

pub fn patchWH(self: VitConfig) [2]u32
{
    return .{self.patch_size, self.patch_size};
}

// getting the number of patches created from the image and patch size configuraations
pub fn numPatches(self: VitConfig) u32 {
    const image_size:[2]u32=self.imageWh();
    const patch_size:[2]u32= self.patchWH();

    return (image_size[0]/patch_size[0])*(image_size[1]/patch_size[1]);
}
};

// making the model struct, fill in later
pub const buffer= zml.Bufferized(Model);

pub const Model = struct{
    embeddings: Embe
};

// ViT Patch embeddings
const PatchEmbed= struct{
    weight: zml.Tensor,
    bias: ?zml.Tensor= null,

    pub fn init (store: zml.io.TensorStore.View, config: VitConfig) !PatchEmbed {
        const patch_size= config.patchWH();
        return .{
            .weight= store.createTensor(subkey: "projection.weight", 
                tagz: .{.dout, .c, .kh, .kw},
                partitioning:.{ .dout = .model, .c = .replicated, .kh = .replicated, .kw = .replicated }),

            .bias= store.createTensor(subkey: "projection.bias", tagz: .{})
        }
    
}