(** Backward compatibility. *)

(* This module contains functions which have been defined in recent version of the standard library, copied here in order to be able to build on the CI of github. You should not need this at home. *)

module Array = struct
  include Array

  let find_index p a =
    let n = length a in
    let rec loop i =
      if i = n then None
      else if p (unsafe_get a i) then Some i
      else loop (succ i) in
    loop 0

  let map_inplace f a =
    for i = 0 to length a - 1 do
      unsafe_set a i (f (unsafe_get a i))
    done
end

module List = struct
  include List

  let take n l =
    let rec aux n l =
      match n, l with
      | 0, _ | _, [] -> []
      | n, x::l -> x::aux (n - 1) l
    in
    if n <= 0 then [] else aux n l
end
