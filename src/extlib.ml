module File = struct
  let read fname =
    In_channel.with_open_bin fname
      (fun ic -> really_input_string ic (in_channel_length ic))
end

module Array = struct
  include Array

  (** Index of an element. *)
  let index a x =
    Option.get @@ Array.find_index (fun y -> x = y) a

  (** Fisher-Yates shuffle of an array. *)
  let shuffle a =
    let n = Array.length a in
    for i = n - 1 downto 1 do
      let j = Random.int (i + 1) in
      let x = a.(i) in
      a.(i) <- a.(j);
      a.(j) <- x
    done
end

module Random = struct
  include Random

  (** Generate a random float along a Gaussian (or normal) distribution. *)
  let gauss () =
    let u1 = Random.float 1.0 in
    let u2 = Random.float 1.0 in
    sqrt (-2. *. log u1) *. cos (2. *. Float.pi *. u2)
end
