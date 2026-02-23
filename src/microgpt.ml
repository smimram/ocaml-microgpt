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
    |> List.shuffle
  in
  (* List.iter print_endline docs; *)
  Printf.printf "num docs: %d\n%!" (List.length docs);

  (* Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back. *)
  let uchars =
    (* unique characters in the dataset become token ids *)
    List.to_seq docs
    |> Seq.concat_map String.to_seq
    |> List.of_seq
    |> List.sort_uniq compare
    |> Array.of_list
  in
  (* token id for a special Beginning of Sequence (BOS) token *)
  let _bos = Array.length uchars in
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
(* state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)} *)
(* for i in range(n_layer): *)
    (* state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd) *)
    (* state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd) *)
    (* state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd) *)
    (* state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd) *)
    (* state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd) *)
    (* state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd) *)
(* params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value] *)
(* print(f"num params: {len(params)}") *)

  
  ignore (Autograd.grad)
