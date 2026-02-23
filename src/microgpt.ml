module File = struct
  let read fname =
    In_channel.with_open_bin fname
      (fun ic -> really_input_string ic (in_channel_length ic))
end

module List = struct
  include List

  let shuffle l =
    l
    |> List.map (fun x -> (Random.bits (), x))
    |> List.sort (fun (a, _) (b, _) -> Stdlib.compare a b)
    |> List.map snd
end

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

  ignore (Autograd.grad)
