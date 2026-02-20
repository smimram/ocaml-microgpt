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
  let docs =
    File.read "input.txt"
    |> String.split_on_char '\n'
    |> List.shuffle
  in
  List.iter print_endline docs
