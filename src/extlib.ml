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

module Random = struct
  include Random

  (** Generate a random float along a Gaussian (or normal) distribution. *)
  let gauss () =
    let u1 = Random.float 1.0 in
    let u2 = Random.float 1.0 in
    sqrt (-2. *. log u1) *. cos (2. *. Float.pi *. u2)
end
