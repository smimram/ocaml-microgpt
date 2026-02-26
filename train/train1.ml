(** Bigram language model with a single-layer MLP, trained by gradient descent. *)

(*
Same as train0.ml:
- Dataset, tokenizer, training loop structure, inference

Different from train0.ml:
- Model is a neural network (MLP) instead of a count table
- Training is gradient descent (SGD) instead of counting
- Introduces: softmax, linear, numerical and analytic gradients

The MLP is effectively a differentiable version of the bigram count table:
token_id -> embedding lookup -> hidden layer -> logits -> softmax -> probs.
The gradient tells us how to nudge each parameter to reduce the loss. We show
two ways to compute it: numerically (perturb and measure) and analytically
(chain rule). They agree, but the analytic version is O(params) faster.
*)

open Extlib

let relu x = max 0. x

module Vector = struct
  type t = float array

  let create n : t = Array.make n 0.

  let dim (v:t) = Array.length v

  let map = Array.map

  let add (v:t) (w:t) : t =
    Array.map2 (+.) v w

  let mul (v:t) (w:t) : t =
    Array.map2 ( *. ) v w

  let cmul x (v:t) : t =
    Array.map (( *. ) x) v

  let sum (v:t) =
    Array.fold_left (+.) 0. v

  let dot (v:t) (w:t) =
    sum @@ mul v w

  let soft_max (logits:t) =
    let max_val = Array.fold_left max min_float logits in
    let exps = Array.map (fun v -> exp (v -. max_val)) logits in
    let total = sum exps in
    Array.map (fun e -> e /. total) exps
end

module Matrix = struct
  type t = float array array

  let init rows cols f =
    Array.init rows (fun i -> Array.init cols (fun j -> f i j))

  let iter2 f (a:t) (b:t) =
    Array.iter2 (Array.iter2 f) a b

  let map f (a:t) : t =
    Array.map (Vector.map f) a

  let coefficients (a:t) =
    List.flatten @@ List.map Array.to_list @@ Array.to_list a

  let ap (a:t) (x:Vector.t) : Vector.t =
    Array.map (fun w -> Vector.dot w x) a

  let cmul x (a:t) : t =
    Array.map (Vector.cmul x) a

  let add (a:t) (b:t) : t =
    Array.map2 Vector.add a b
end

type state =
  {
    mutable wte : Matrix.t;
    mutable mlp_fc1 : Matrix.t;
    mutable mlp_fc2 : Matrix.t;
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
  let matrix nout nin = Matrix.init nout nin (fun _ _ -> 0.08 *. Random.gauss ()) in
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

  (* Forward pass: run the model on a token sequence, return the average loss *)
  let forward tokens n =
    let loss = ref 0. in
    for pos_id = 0 to n - 1 do
      let token_id = tokens.(pos_id) in
      let target_id = tokens.(pos_id + 1) in
      let logits = mlp token_id in
      let probs = Vector.soft_max logits in
      let loss_t = -. (log probs.(target_id)) in
      loss := !loss +. loss_t
    done;
    !loss /. float n
  in

  (* Two ways to compute the gradient of the loss w.r.t. all parameters: *)

  (* Perturb each parameter by eps, measure change in loss. *)
  let numerical_gradient tokens n =
    let loss = forward tokens n in
    let eps = 1e-5 in
    let grad mat =
      Array.map (fun row ->
          Array.mapi (fun j x ->
              row.(j) <- x +. eps;
              let loss_plus = forward tokens n in
              row.(j) <- x;
              (loss_plus -. loss) /. eps
            ) row
        ) mat
    in
    let grad =
      {
        wte = grad state.wte;
        mlp_fc1 = grad state.mlp_fc1;
        mlp_fc2 = grad state.mlp_fc2;
      }
    in
    loss, grad
  in

  (* Derive the gradient analytically using the chain rule. *)
  let analytic_gradient tokens n =
    let grad =
      let zero _ = 0. in
      {
        wte = Matrix.map zero state.wte;
        mlp_fc1 = Matrix.map zero state.mlp_fc1;
        mlp_fc2 = Matrix.map zero state.mlp_fc2;
      }
    in
    let loss = ref 0. in
    for pos_id = 0 to n - 1 do
      let token_id = tokens.(pos_id) in
      let target_id = tokens.(pos_id+1) in
      (* forward pass (saving intermediates for backward) *)
      let x = state.wte.(token_id) in
      let h_pre = Matrix.ap state.mlp_fc1 x in
      let h = Vector.map relu h_pre in
      let logits = Matrix.ap state.mlp_fc2 h in
      let probs = Vector.soft_max logits in
      let loss_t = -.(log probs.(target_id)) in
      loss := !loss +. loss_t;
      (* backward pass: chain rule, layer by layer *)
      (* d(loss)/d(logits) for softmax + cross-entropy = probs - one_hot(target) *)
      let dlogits = Vector.cmul (1. /. float n) probs in
      dlogits.(target_id) <- dlogits.(target_id) -. 1. /. float n;
      (* d(loss)/d(mlp_fc2), d(loss)/d(h): logits = mlp_fc2 @ h *)
      let dh = Vector.create @@ Vector.dim h in
      for i = 0 to Vector.dim dlogits - 1 do
        for j = 0 to Vector.dim h - 1 do
          grad.mlp_fc2.(i).(j) <- grad.mlp_fc2.(i).(j) +. dlogits.(i) *. h.(j);
          dh.(j) <- dh.(j) +. state.mlp_fc2.(i).(j) *. dlogits.(i)
        done
      done;
      (* d(loss)/d(h_pre): relu backward *)
      let dh_pre = Vector.mul dh @@ Vector.map (fun x -> if x > 0. then 1. else 0.) h_pre in
      (* d(loss)/d(mlp_fc1), d(loss)/d(x): h_pre = mlp_fc1 @ x *)
      let dx = Vector.create @@ Vector.dim x in
      for i = 0 to Vector.dim dh_pre - 1 do
        for j = 0 to Vector.dim x - 1 do
          grad.mlp_fc1.(i).(j) <- grad.mlp_fc1.(i).(j) +. dh_pre.(i) *. x.(j);
          dx.(j) <- dx.(j) +. state.mlp_fc1.(i).(j) *. dh_pre.(i)
        done
      done;
      (* d(loss)/d(wte[token_id]): x = wte[token_id] *)
      for j = 0 to Vector.dim x - 1 do
        grad.wte.(token_id).(j) <- grad.wte.(token_id).(j) +. dx.(j)
      done;
    done;
    let loss = !loss /. float n in
    loss, grad
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

    (* Gradient check: verify numerical and analytic gradients agree *)
    if step = 0 then
      (
        let loss_n, grad_n = numerical_gradient tokens n in
        let loss_a, grad_a = analytic_gradient tokens n in
        let grad_diff =
          let ans = ref 0. in
          let f = Matrix.iter2 (fun gn ga -> ans := max !ans (abs_float (gn -. ga))) in
          f grad_n.wte grad_a.wte;
          f grad_n.mlp_fc1 grad_a.mlp_fc1;
          f grad_n.mlp_fc2 grad_a.mlp_fc2;
          !ans
        in
        Printf.printf "gradient check | loss_n %.6f | loss_a %.6f | max diff %.8f\n" loss_n loss_a grad_diff
      );

    (* let loss, grad = numerical_gradient tokens n in *)
    let loss, grad = analytic_gradient tokens n in

    (* SGD update *)
    let lr_t = learning_rate *. (1. -. float step /. float num_steps) in (* linear learning rate decay *)

    state.wte <- Matrix.add state.wte @@ Matrix.cmul (-.lr_t) grad.wte;
    state.mlp_fc1 <- Matrix.add state.mlp_fc1 @@ Matrix.cmul (-.lr_t) grad.mlp_fc1;
    state.mlp_fc2 <- Matrix.add state.mlp_fc2 @@ Matrix.cmul (-.lr_t) grad.mlp_fc2;

    if step < 5 || step mod 100 = 0 then
      Printf.printf "step %4d / %4d | loss %.4f\n%!" step num_steps loss
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
      let probs = Vector.soft_max @@ Vector.cmul (1. /. temperature) logits in
      token_id := Random.index @@ probs;
      if !token_id = bos then pos_id := block_size
      else sample := uchars.(!token_id) :: !sample;
    done;
    let sample = String.of_seq @@ List.to_seq @@ List.rev !sample in
    Printf.printf "sample %2d: %s\n%!" (sample_idx+1) sample
  done
