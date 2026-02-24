(** Bigram language model trained by counting. *)

(*
 The structure is identical to train.py: tokenize, forward pass, update, sample.
The only difference is what's inside the "model" box:
- train.py: gpt(token_id) -> logits, trained by gradient descent
- train0.py: bigram(token_id) -> probs, trained by counting

A bigram model is a special case of a GPT where there is no attention (each token
only looks at itself), no MLP, and the "embedding" is just a row in a lookup table.
Counting is the closed-form solution for this case; gradient descent is what you
need when the model is too expressive for exact solutions.
 *)

open Extlib

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

  (* Initialize the parameters: a bigram count table. state_dict[i][j] = how many times token j follows token i. *)
  let state = Array.make_matrix vocab_size vocab_size 0 in

  (* The "model": given a token_id, return the probability distribution over the next token *)
  let bigram token_id =
    let row = state.(token_id) in
    (* add-one (Laplace) smoothing *)
    let row = Array.map succ row in
    let total = Array.fold_left (+) 0 row  in
    Array.map (fun c -> float c /. float total) row
  in

  (* Train the model *)
  let num_steps = 1000 in
  for step = 0 to num_steps - 1 do
    (* Take single document, tokenize it, surround it with BOS special token on both sides *)
    let tokens =
      docs.(step mod Array.length docs)
      |> String.to_seq
      |> Seq.map (Array.index uchars)
      |> (fun doc -> Seq.concat (List.to_seq [Seq.return bos; doc; Seq.return bos]))
      |> Array.of_seq
    in
    let n = Array.length tokens - 1 in

    (* Forward pass: compute the loss for this document *)
    let loss = ref 0. in
    for pos_id = 0 to n - 1 do
      let token_id = tokens.(pos_id) in
      let target_id = tokens.(pos_id + 1) in
      let probs = bigram token_id in
      let loss_t = -.(log probs.(target_id)) in
      loss := !loss +. loss_t
    done;
    let loss = !loss /. float n in

    (* Update the model: incorporate this document's bigram counts *)
    for pos_id = 0 to n - 1 do
      let token_id = tokens.(pos_id) in
      let target_id = tokens.(pos_id + 1) in
      state.(token_id).(target_id) <- state.(token_id).(target_id) + 1
    done;

    Printf.printf "step %4d / %4d | loss %.4f\r" (step+1) num_steps loss
  done;

  (* Inference: sample new names from the model *)
  print_endline "\n--- inference (new, hallucinated names) ---";
  let block_size = 16 in (* maximum sequence length *)
  for sample_idx = 0 to 20 - 1 do
    let token_id = bos in
    let sample = ref [] in
    let pos_id = ref 0 in
    while !pos_id < block_size do
      incr pos_id;
      let token_id = Random.index @@ bigram token_id in      
      if token_id = bos then pos_id := block_size
      else sample := uchars.(token_id) :: !sample;
    done;
    let sample = String.of_seq @@ List.to_seq @@ List.rev !sample in
    Printf.printf "sample %2d: %s\n%!" (sample_idx+1) sample
  done
