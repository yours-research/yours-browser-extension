// Slot 0 is root. Parent and token arrays include its ignored entry.
@group(0) @binding(0) var<uniform> slots: u32;
@group(0) @binding(1) var<storage,read> parents: array<u32>;
@group(0) @binding(2) var<storage,read> tokens: array<u32>;
@group(0) @binding(3) var<storage,read> argmax: array<u32>;
@group(0) @binding(4) var<storage,read_write> choice: array<u32>;
@compute @workgroup_size(1)
fn main() {
  var leaf = 0u;
  var accepted = 0u;
  choice[2] = 0u;
  for (var i = 1u; i < slots; i++) {
    if (parents[i] == leaf && tokens[i] == argmax[leaf]) {
      leaf = i;
      accepted++;
      choice[2u + accepted] = i;
    }
  }
  choice[0] = accepted;
  choice[1] = argmax[leaf];
}
