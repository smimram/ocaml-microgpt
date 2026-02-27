open Extlib
open Autograd

type layer =
  {
    attn_wq : Matrix.t; (** query weights *)
    attn_wk : Matrix.t; (** key weights *)
    attn_wv : Matrix.t; (** value weights *)
    attn_wo : Matrix.t; (** output weights *)
    mlp_fc1 : Matrix.t;
    mlp_fc2 : Matrix.t;
  }

type state =
  {
    wte : Matrix.t; (** term embedding *)
    wpe : Matrix.t; (** position embedding *)
    layer : layer array;
    lm_head : Matrix.t;
  }

let () =
  Random.self_init ();

  if not (Sys.file_exists "input.txt") then File.download "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt" "input.txt";
  let doc = File.read "input.txt" in
  Printf.printf "doc length: %d\n%!" (String.length doc);

  (* unique characters in the dataset *)
  let uchars =
    doc
    |> String.to_seq
    |> List.of_seq
    |> List.sort_uniq compare
    |> Array.of_list
  in
  (* total number of unique tokens *)
  let vocab_size = Array.length uchars in
  Printf.printf "vocab size: %d\n%!" vocab_size;

  (* Initialize the parameters, to store the knowledge of the model *)
  let n_layer = 1 in (* depth of the transformer neural network (number of layers) *)
  let n_embd = 16 in (* width of the network (embedding dimension) *)
  let block_size = 16 in (* maximum context length of the attention window *)
  let n_head = 4 in (* number of attention heads *)
  let head_dim = n_embd / n_head in (* derived dimension of each head *)
  let matrix ?(std=0.08) nout nin = Matrix.init nout nin (fun _ _ -> const (std *. Random.gauss ())) in
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
            mlp_fc1 = matrix (4*n_embd) n_embd;
            mlp_fc2 = matrix n_embd (4*n_embd);
          }
        )
    in
    {
      wte = matrix vocab_size n_embd;
      wpe = matrix block_size n_embd;
      layer;
      lm_head = matrix vocab_size n_embd;
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

  (* Define the model architecture *)
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
      let q = Matrix.ap state.layer.(li).attn_wq !x in
      let k = Matrix.ap state.layer.(li).attn_wk !x in
      let v = Matrix.ap state.layer.(li).attn_wv !x in
      keys.(li) <- List.take block_size (k :: keys.(li));
      values.(li) <- List.take block_size (v :: values.(li));
      let x_attn = ref [] in
      for h = 0 to n_head - 1 do
        let hs = h * head_dim in
        let sub v = Vector.subvector v hs head_dim in
        let q_h = sub q in
        let k_h = List.map sub keys.(li) in
        let v_h = List.map sub values.(li) in
        let attn_logits = List.map (fun k_hi -> cmul (1. /. sqrt (float head_dim)) (Vector.dot q_h k_hi)) k_h |> Array.of_list in
        let attn_weights = Vector.soft_max attn_logits in
        let head_out = Array.map (Vector.dot attn_weights) (Matrix.transpose @@ Array.of_list v_h) in
        x_attn := head_out :: !x_attn;
      done;
      let x_attn = Array.concat @@ List.rev !x_attn in
      x := Matrix.ap state.layer.(li).attn_wo x_attn;
      x := Vector.add !x x_residual;

      (* 2) MLP block *)
      let x_residual = !x in
      x := Vector.rms_norm !x;
      x := Matrix.ap state.layer.(li).mlp_fc1 !x;
      x := Vector.map relu !x;
      x := Matrix.ap state.layer.(li).mlp_fc2 !x;
      x := Vector.add !x x_residual
    done;
    Matrix.ap state.lm_head !x
  in

  (* Let there be Adam, the blessed optimizer and its buffers *)
  let learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8 in
  let m = Array.make (List.length params) 0. in (* first moment buffer *)
  let v = Array.make (List.length params) 0. in (* second moment buffer *)

  (* Repeat in sequence *)
  let num_steps = String.length doc / block_size in (* number of training steps *)
  (* let num_steps = 500 in *)
  let t0 = Unix.time () in
  for step = 0 to num_steps - 1 do

    let tokens =
      String.sub doc (step * block_size) block_size
      |> String.to_seq
      |> Seq.map (Array.index uchars)
      |> Array.of_seq
    in
    let n = min block_size (Array.length tokens - 1) in

    (* Forward the token sequence through the model, building up the computation graph all the way to the loss *)
    let keys = Array.make n_layer [] in
    let values = Array.make n_layer [] in
    let loss = ref (const 0.) in
    for pos_id = 0 to n - 1 do
      let token_id = tokens.(pos_id) in
      let target_id = tokens.(pos_id + 1) in
      let logits = gpt token_id pos_id keys values in
      let probs = Vector.soft_max logits in
      let loss_t = neg (log probs.(target_id)) in
      loss := add !loss loss_t
    done;
    (* final average loss over the document sequence *)
    let loss = cmul (1. /. float n) !loss in

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

    let t = Unix.time () in
    let t = (t -. t0) *. (float num_steps -. float step) /. float step in
    let t = int_of_float t in
    if step mod 10 = 0 then
      Printf.printf "step %4d / %4d | loss %.4f | eta %02d:%02d\r%!" step num_steps (value loss) (t/60) (t mod 60)
  done;
  print_newline ();

  (* Inference: may the model babble back to us *)
  let temperature = 0.5 in (* in (0, 1], control the "creativity" of generated text, low to high *)
  print_endline "--- inference ---";
  let keys = Array.make n_layer [] in
  let values = Array.make n_layer [] in
  let token_id = ref @@ Random.int @@ Array.length uchars in
  let pos_id = ref 0 in
  while !pos_id < 1000 do
    let logits = gpt !token_id (!pos_id mod block_size) keys values in
    let probs = Vector.soft_max @@ Vector.cmul (const (1. /. temperature)) logits in
    token_id := Random.index @@ Array.map value probs;
    output_char stdout uchars.(!token_id);
    incr pos_id
  done;
  print_newline ()
