(** Test extlib. *)

open Extlib

let () =
  Random.self_init ();
  let a = [| 0.1; 0.8; 0.1 |] in 
  for _ = 1 to 10 do
    Printf.printf "%d\n" @@ Random.index a
  done
