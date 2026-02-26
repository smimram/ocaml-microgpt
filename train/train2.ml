(** Bigram language model with a single-layer MLP, trained with autograd. *)

(*
Same as train1.ml:
- Dataset, tokenizer, model architecture (MLP), SGD optimizer, inference

Different from train1.ml:
- Introduces autograd to compute gradients automatically
- No more manual analytic_gradient or numerical_gradient
- The forward pass builds a computation graph, loss.backward() traverses it

The hand-derived chain rule from train1.py is now automated: each operation
records its local gradient, and backward() applies the chain rule recursively.
The training loop becomes: forward pass -> loss.backward() -> SGD update.
 *)

open Extlib
open Autograd

type state =
  {
    wte : Matrix.t;
    mlp_fc1 : Matrix.t;
    mlp_fc2 : Matrix.t;
  }

let () =
  Random.self_init ();

  (* Let there be a Dataset docs: a list of documents (e.g. a list of names). *)
  if not (Sys.file_exists "input.txt") then File.download "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt" "input.txt";
  let docs =
    File.read "input.txt"
    |> String.split_on_char '\n'
    |> List.map String.trim
    |> List.filter (fun s -> s <> "")
    |> Array.of_list
  in
  Array.shuffle docs;

  (* Tokenizer: character-level, with a special BOS (Beginning of Sequence) token *)
  let uchars =
    (* unique characters in the dataset become token ids *)
    Array.to_seq docs
    |> Seq.concat_map String.to_seq
    |> List.of_seq
    |> List.sort_uniq compare
    |> Array.of_list
  in
  (* token id for a special Beginning of Sequence (BOS) token *)
  let bos = Array.length uchars in
  (* total number of unique tokens, +1 is for BOS *)
  let vocab_size = Array.length uchars + 1 in
  Printf.printf "vocab size: %d\n%!" vocab_size;

  (* Initialize the parameters *)
  let n_embd = 16 in (* embedding dimension *)
  let matrix nout nin = Matrix.init nout nin (fun _ _ -> const (0.08 *. Random.gauss ())) in
  let state =
    {
      wte = matrix vocab_size n_embd;
      mlp_fc1 = matrix (4 * n_embd) n_embd;
      mlp_fc2 = matrix vocab_size (4 * n_embd);
    }
  in
  let params = List.flatten @@ List.map Matrix.coefficients [state.wte; state.mlp_fc1; state.mlp_fc2] in
  Printf.printf "num params: %d\n" (List.length params);

  let mlp token_id =
    state.wte.(token_id)
    |> Matrix.ap state.mlp_fc1
    |> Vector.map relu
    |> Matrix.ap state.mlp_fc2
  in

  (* Train the model *)
  let num_steps = 1000 in
  let learning_rate = 1. in
  for step = 0 to num_steps do

    (* Take single document, tokenize it, surround it with BOS special token on both sides *)
    let tokens =
      docs.(step mod Array.length docs)
      |> String.to_seq
      |> Seq.map (Array.index uchars)
      |> (fun doc -> Seq.concat (List.to_seq [Seq.return bos; doc; Seq.return bos]))
      |> Array.of_seq
    in
    let n = Array.length tokens - 1 in

    (* Forward pass *)
    let loss = ref @@ const 0. in
    for pos_id = 0 to n - 1 do
      let token_id = tokens.(pos_id) in
      let target_id = tokens.(pos_id + 1) in
      let logits = mlp token_id in
      let probs = Vector.soft_max logits in
      let loss_t = neg (log probs.(target_id)) in
      loss := add !loss loss_t
    done;
    let loss = cmul (1. /. float n) !loss in

    (* Backward pass *)
    backward loss;

    (* SGD update *)
    let lr_t = learning_rate *. (1. -. float step /. float num_steps) in (* linear learning rate decay *)
    List.iter (fun p ->
        p.value <- value p -. lr_t *. grad p;
        p.grad <- 0.
      ) params;

    if step < 5 || step mod 100 = 0 then
    Printf.printf "step %4d / %4d | loss %.4f\n%!" step num_steps (value loss)
  done;

  (* Inference: sample new names from the model *)
  print_endline "--- inference (new, hallucinated names) ---";
  let temperature = 0.5 in
  let block_size = 16 in (* maximum sequence length *)
  for sample_idx = 0 to 20 - 1 do
    let token_id = ref bos in
    let sample = ref [] in
    let pos_id = ref 0 in
    while !pos_id < block_size do
      incr pos_id;
      let logits = mlp !token_id in
      let probs = Vector.soft_max @@ Vector.cmul (const (1. /. temperature)) logits in
      token_id := Random.index @@ Array.map value @@ probs;
      if !token_id = bos then pos_id := block_size
      else sample := uchars.(!token_id) :: !sample;
    done;
    let sample = String.of_seq @@ List.to_seq @@ List.rev !sample in
    Printf.printf "sample %2d: %s\n%!" (sample_idx+1) sample
  done
