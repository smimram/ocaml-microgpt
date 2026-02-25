(** Extended standard library. *)

module File = struct
  let read fname =
    In_channel.with_open_bin fname
      (fun ic -> really_input_string ic (in_channel_length ic))

  let download url fname =
    let ans = Sys.command @@ Filename.quote_command "wget" [url;"-O";fname] in
    assert (ans = 0)
end

module Array = struct
  include Array

  (* Backward compabtibility. *)
  let find_index p a =
    let n = length a in
    let rec loop i =
      if i = n then None
      else if p (unsafe_get a i) then Some i
      else loop (succ i) in
    loop 0

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

  (** Pick an index in a list of weights. *)
  let index a =
    let total = Array.fold_left ( +. ) 0. a in
    let r = ref @@ Random.float total in
    Option.value ~default:(Array.length a - 1) @@ Array.find_index (fun w -> if !r < w then true else (r := !r -. w; false)) a
end
