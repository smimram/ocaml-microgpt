open Extlib
open Autograd

type layer =
  {
    attn_wq : Matrix.t;
    attn_wk : Matrix.t;
    attn_wv : Matrix.t;
    attn_wo : Matrix.t;
    mlp_fc1 : Matrix.t;
    mlp_fc2 : Matrix.t;
  }

type state =
  {
    wte : Matrix.t;
    wpe : Matrix.t;
    lm_head : Matrix.t;
    layer : layer array;
  }

let () =
  Random.self_init ();

  (* Let there be a Dataset docs: a list of documents (e.g. a list of names). *)
  let docs =
    File.read "input.txt"
    |> String.split_on_char '\n'
    |> List.map String.trim
    |> List.filter (fun s -> s <> "")
    |> Array.of_list
  in
  Array.shuffle docs;
  (* List.iter print_endline docs; *)
  Printf.printf "num docs: %d\n%!" (Array.length docs);

  (* Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back. *)
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

  (* Initialize the parameters, to store the knowledge of the model *)
  let n_layer = 1 in (* depth of the transformer neural network (number of layers) *)
  let n_embd = 16 in (* width of the network (embedding dimension) *)
  let block_size = 16 in (* maximum context length of the attention window (note: the longest name is 15 characters) *)
  let n_head = 4 in (* number of attention heads *)
  let head_dim = n_embd / n_head in (* derived dimension of each head *)
  let matrix ?(std=0.08) nout nin =
    Matrix.init nout nin (fun _ _ -> const (std *. Random.gauss ()))
  in
  let state =
    let layer =
      Array.init
        n_layer
        (fun _ ->
          {
            attn_wq = matrix n_embd n_embd;
            attn_wk = matrix n_embd n_embd;
            attn_wv = matrix n_embd n_embd;
            attn_wo = matrix n_embd n_embd;
            mlp_fc1 = matrix (4 * n_embd) n_embd;
            mlp_fc2 = matrix n_embd (4 * n_embd);
          }
        )
    in
    {
      wte = matrix vocab_size n_embd;
      wpe = matrix block_size n_embd;
      lm_head = matrix vocab_size n_embd;
      layer;
    }
  in
  let params =
    let layers =
      Array.to_list state.layer
      |> List.map (fun l -> [l.attn_wq; l.attn_wk; l.attn_wv; l.attn_wo; l.mlp_fc1; l.mlp_fc2])
      |> List.flatten
    in
    List.flatten @@ List.map Matrix.coefficients ([state.wte; state.wpe; state.lm_head]@layers)
  in
  Printf.printf "num params: %d\n%!" (List.length params);

  (* Define the model architecture: a function mapping tokens and parameters to logits over what comes next
     Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
   *)
  let linear x w = Matrix.ap w x in
  let gpt token_id pos_id keys values =
    (* token embedding *)
    let tok_emb = state.wte.(token_id) in
    (* position embedding *)
    let pos_emb = state.wpe.(pos_id) in
    (* joint token and position embedding *)
    let x = Vector.add tok_emb pos_emb in
    (* note: not redundant due to backward pass via the residual connection *)
    let x = Vector.rms_norm x in
    let x = ref x in
    for li = 0 to n_layer - 1 do
      (* 1) Multi-head Attention block *)
      let x_residual = !x in
      x := Vector.rms_norm !x;
      let q = linear !x state.layer.(li).attn_wq in
      let k = linear !x state.layer.(li).attn_wk in
      let v = linear !x state.layer.(li).attn_wv in
      keys.(li) <- Array.append keys.(li) [|k|];
      values.(li) <- Array.append values.(li) [|v|];
      let x_attn = ref [||] in
      for h = 0 to n_head - 1 do
        let hs = h * head_dim in
        let sub v = Vector.sub v hs head_dim in
        let q_h = sub q in
        let k_h = Array.map sub keys.(li) in
        let v_h = Array.map sub values.(li) in
        let attn_logits = Vector.init (Array.length k_h) (fun t -> div (Vector.dot q_h k_h.(t)) (const (float head_dim ** 0.5))) in
        let attn_weights = Vector.soft_max attn_logits in
        let head_out = Array.init head_dim (fun j -> Vector.dot attn_weights (Matrix.transpose v_h).(j)) in
        x_attn := Array.append !x_attn head_out;
      done;
      x := linear !x_attn state.layer.(li).attn_wo;
      x := Vector.add !x x_residual;

      (* 2) MLP block *)
      let x_residual = !x in
      x := Vector.rms_norm !x;
      x := linear !x state.layer.(li).mlp_fc1;
      x := Vector.map relu !x;
      x := linear !x state.layer.(li).mlp_fc2;
      x := Vector.add !x x_residual
    done;
    linear !x state.lm_head
  in

  (* Let there be Adam, the blessed optimizer and its buffers *)
  let learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8 in
  let m = Array.make (List.length params) 0. in (* first moment buffer *)
  let v = Array.make (List.length params) 0. in (* second moment buffer *)

  (* Repeat in sequence *)
  let num_steps = 1000 in (* number of training steps *)
  for step = 0 to num_steps - 1 do

    (* Take single document, tokenize it, surround it with BOS special token on both sides *)
    let tokens =
      docs.(step mod Array.length docs)
      |> String.to_seq
      |> Seq.map (Array.index uchars)
      |> (fun doc -> Seq.concat (List.to_seq [Seq.return bos; doc; Seq.return bos]))
      |> Array.of_seq
    in
    let n = min block_size (Array.length tokens - 1) in

    (* Forward the token sequence through the model, building up the computation graph all the way to the loss *)
    let keys = Array.init n_layer (fun _ -> [||]) in
    let values = Array.init n_layer (fun _ -> [||]) in
    let losses = ref [||] in
    for pos_id = 0 to n - 1 do
      let token_id = tokens.(pos_id) in
      let target_id = tokens.(pos_id + 1) in
      let logits = gpt token_id pos_id keys values in
      let probs = Vector.soft_max logits in
      let loss_t = neg (log probs.(target_id)) in
      losses := Array.append !losses [|loss_t|]
    done;
    (* final average loss over the document sequence. May yours be low. *)
    let loss = mul (const (1. /. float n)) (Vector.sum !losses) in

    (* Backward the loss, calculating the gradients with respect to all model parameters *)
    backward loss;

    (* Adam optimizer update: update the model parameters based on the corresponding gradients *)
    let lr_t = learning_rate *. (1. -. float step /. float num_steps) in (* linear learning rate decay *)
    List.iteri
      (fun i p ->
        m.(i) <- beta1 *. m.(i) +. (1. -. beta1) *. p.grad;
        v.(i) <- beta2 *. v.(i) +. (1. -. beta2) *. p.grad ** 2.;
        let m_hat = m.(i) /. (1. -. beta1 ** (float (step + 1))) in
        let v_hat = v.(i) /. (1. -. beta2 ** (float (step + 1))) in
        p.value <- p.value -. lr_t *. m_hat /. (v_hat ** 0.5 +. eps_adam);
        p.grad <- 0.
      ) params;

    Printf.printf "step %4d / %4d | loss %.4f\r%!" (step+1) num_steps (value loss)
  done;

  (* Inference: may the model babble back to us *)
  let temperature = 0.5 in (* in (0, 1], control the "creativity" of generated text, low to high *)
  print_string "\n--- inference (new, hallucinated names) ---";
  for sample_idx = 0 to 19 do
    let keys = Array.make n_layer [||] in
    let values = Array.make n_layer [||] in
    let token_id = bos in
    let sample = ref [] in
    let pos_id = ref 0 in
    while !pos_id < block_size do
      incr pos_id;
      let logits = gpt token_id !pos_id keys values in
      let probs = Vector.soft_max @@ Vector.cmul (const (1. /. temperature)) logits in
      let token_id =
        let probs = List.mapi (fun i x -> value x, i) (Array.to_list probs) in
        Random.element probs
      in
      if token_id = bos then pos_id := block_size
      else sample := uchars.(token_id) :: !sample
    done;
    let sample = String.of_seq @@ List.to_seq @@ List.rev !sample in
    Printf.printf "sample %02d: %s\n%!" (sample_idx+1) sample
  done


